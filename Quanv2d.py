import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from typing import  Union, List, Iterator

import qiskit as qk
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from torch_connector import TorchConnector


class Quanv2d(nn.Module):

    '''
        A quantum convolutional layer
        args
            input_channel: number of input channels
            output_channel: number of output channels
            num_qubits: number of qubits
            num_weight: number of weights
            kernel_size: size of the kernel
            stride: stride of the kernel
    '''
    def __init__(self,
                 input_channel : int,
                 output_channel : int,
                 num_qubits : int,
                 num_weight : int, 
                 kernel_size : int = 3, 
                 stride : int = 1
                 ):

        super(Quanv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.num_weight = num_weight
        self.num_input = kernel_size * kernel_size * input_channel
        self.num_qubits = num_qubits
        self.qnn = TorchConnector(self.Sampler())
        assert 2**num_qubits >= output_channel, '2**num_qubits must be greater than output_channel'

    def build_circuit(self,
                num_weights : int,
                num_input : int,
                num_qubits : int = 3
                ) -> tuple[QuantumCircuit, Iterator[qk.circuit.Parameter], Iterator[qk.circuit.Parameter]]:
        '''
        build the quantum circuit
        param
            num_weights: number of weights
            num_input: number of inputs
            num_qubits: number of qubits
        return
            qc: quantum circuit
            weight_params: weight parameters
            input_params: input parameters
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
        return qc, weight_params, input_params
    
    def Sampler(self) -> SamplerQNN:
        '''
        build the quantum circuit
        param
            num_weights: number of weights
            num_input: number of inputs
            num_qubits: number of qubits
        return
            qc: quantum circuit
        '''
        qc,weight_params,input_params = self.build_circuit(self.num_weight,self.num_input,3)
        
        #use SamplerQNN to convert the quantum circuit to a PyTorch module
        qnn = SamplerQNN(
                        circuit = qc,
                        weight_params = weight_params,
                        interpret=self.interpret, 
                        input_params=input_params,
                        output_shape=self.output_channel,
                         )
        return qnn

    def interpret(self, X: Union[List[int],int]) -> Union[int,List[int]]:
        '''
        interpret the output of the quantum circuit using the modulo function
        this function is used in SamplerQNN
        args
            X: output of the quantum circuit
        return
            the remainder of the output divided by the number of output channels
        '''
        return X % self.output_channel

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        '''
        forward function for the quantum convolutional layer
        args
            X: input tensor with shape (batch_size, input_channel, height, width)
        return
            X: output tensor with shape (batch_size, output_channel, height, width)
        '''
        height = len(range(0,X.shape[2]-self.kernel_size+1,self.stride))
        width = len(range(0,X.shape[3]-self.kernel_size+1,self.stride))
        output = torch.zeros((X.shape[0],self.output_channel,height,width))
        X = F.unfold(X[:, :, :, :], kernel_size=self.kernel_size, stride=self.stride)
        qnn_output = self.qnn(X.permute(2, 0, 1)).permute(1, 2, 0)
        qnn_output = torch.reshape(qnn_output,shape=(X.shape[0],self.output_channel,height,width))
        output += qnn_output
        return output
 
if __name__ == '__main__':
    # Define the model
    model = Quanv2d(3, 2, 3, 5,stride=1)
    X = torch.rand((5,3,8,8))
    time0 = time.time()
    X1=model.forward(X)
    time1 = time.time()
    print(time1-time0)
    print(model)
    print(X1.shape)
    print(X1[0])
