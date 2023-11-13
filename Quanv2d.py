import torch
import qiskit as qk
from qiskit import QuantumCircuit
import torch.nn as nn
from qiskit_machine_learning.neural_networks import SamplerQNN
from torch_connector import TorchConnector
import torch.nn.functional as F


# Define the quantum circuit

class Quanv2d(nn.Module):
    def __init__(self,input_channel,output_channel,num_qubits,num_weight, kernel_size=3, stride=1):
        '''
        A quantum convolutional layer
        param 
            input_channel: number of input channels
            output_channel: number of output channels
            num_qubits: number of qubits
            num_weight: number of weights
            kernel_size: size of the kernel
            stride: stride of the kernel
        '''
        super(Quanv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.qnn = TorchConnector(self.Sampler(num_weight,kernel_size * kernel_size * input_channel, num_qubits))
        #check if 2**num_qubits is greater than output_channel
        assert 2**num_qubits >= output_channel, '2**num_qubits must be greater than output_channel'

    def Sampler(self, num_weights, num_input, num_qubits = 3):
        '''
        build the quantum circuit
        param
            num_weights: number of weights
            num_input: number of inputs
            num_qubits: number of qubits
        return
            qc: quantum circuit
        '''
        qc = QuantumCircuit(num_qubits)
        weight_params = [qk.circuit.Parameter('w{}'.format(i)) for i in range(num_weights)]
        input_params = [qk.circuit.Parameter('x{}'.format(i)) for i in range(num_input)]
        #construct the quantum circuit with the parameters
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_input):
            qc.ry(input_params[i]*2*torch.pi, i%num_qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        for i in range(num_weights):
            qc.rx(weight_params[i]*2*torch.pi, i%num_qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

        #use SamplerQNN to convert the quantum circuit to a PyTorch module
        qnn = SamplerQNN(circuit = qc,weight_params = weight_params,interpret=self.interpret, input_params=input_params,output_shape=self.output_channel)
        return qnn

    def interpret(self, X):
        '''
        interpret the output of the quantum circuit using the modulo function
        this function is used in SamplerQNN
        param
            X: output of the quantum circuit
        return
            the remainder of the output divided by the number of output channels
        '''
        return X%self.output_channel

    def forward(self, X):
        '''
        param
            X: input tensor with shape (batch_size, input_channel, height, width)
        return
            X: output tensor with shape (batch_size, output_channel, height, width)
        '''
        #for each input channel we have a quantum circuit to process it
        #and then we add them together
        #get the height and width of the output tensor
        height = len(range(0,X.shape[2]-self.kernel_size+1,self.stride))
        width = len(range(0,X.shape[3]-self.kernel_size+1,self.stride))
        output = torch.zeros((X.shape[0],self.output_channel,height,width))
            
        X = F.unfold(X[:, :, :, :], kernel_size=self.kernel_size, stride=self.stride)
        # print(X.shape)
        qnn_output = self.qnn(X.permute(2, 0, 1)).permute(1, 2, 0)
        qnn_output = torch.reshape(qnn_output,shape=(X.shape[0],self.output_channel,height,width))
        output += qnn_output
        return output
    
        
    
    # def forward2(self,X):
    #     #use for loop to implement the convolution
    #     '''
    #     param
    #         X: input tensor with shape (batch_size, input_channel, height, width)
    #     return
    #         X: output tensor with shape (batch_size, output_channel, height, width)
    #     '''
    #     #for each input channel we have a quantum circuit to process it
    #     #and then we add them together
    #     #get the height and width of the output tensor
    #     height = len(range(0,X.shape[2]-self.kernel_size+1,self.stride))
    #     width = len(range(0,X.shape[3]-self.kernel_size+1,self.stride))
    #     output = torch.zeros((X.shape[0],self.output_channel,height,width))
    #     for i in range(self.input_channel):
    #         #for each kernel we have a quantum circuit to process it
    #         output_temp = torch.zeros((X.shape[0],self.output_channel,height,width))
    #         for jj,j in enumerate(range(0,X.shape[2]-self.kernel_size+1,self.stride)):
    #             for kk,k in enumerate(range(0,X.shape[3]-self.kernel_size+1,self.stride)):
    #                 '''
    #                 #get the kernel
    #                 kernel = X[:,i,j:j+self.kernel_size,k:k+self.kernel_size]
    #                 # print(kernel)
    #                 kernel = torch.reshape(kernel,shape=(X.shape[0],self.kernel_size**2))
    #                 # print(kernel)
    #                 #process the kernel with the quantum circuit
    #                 qnn_output = self.qnn[i](kernel)
    #                 qnn_output = torch.transpose(qnn_output,0,1)
    #                 # print(qnn_output.shape)
    #                 for l in range(self.output_channel):
    #                     output_temp[:,l,jj,kk] = qnn_output[l]
    #                 與下方等價
    #                 '''
    #                 kernel = X[:, i, j:j+self.kernel_size, k:k+self.kernel_size].reshape(X.shape[0], -1)
    #                 qnn_output = self.qnn[i](kernel)
    #                 output_temp[:, :, jj, kk] = qnn_output
    #         #add the output of each input channel together
    #         output += output_temp
    #     return output            
    
    
        

if __name__ == '__main__':
    import time
    # Define the model
    model = Quanv2d(3, 2, 3, 5,stride=1)
    X = torch.rand((5,3,16,16))
    time0 = time.time()
    X1=model.forward(X)
    time1 = time.time()
    # X2 = model.forward2(X)
    # time2 = time.time()
    # assert torch.all(torch.eq(X1,X2)) ; 'forward function is wrong'
    # print(time1-time0,time2-time1)
    print(model)
    print(X1.shape)
    print(X1[0])
