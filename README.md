# Hackathon-NCCU-2023

Hackathon NCCU 2023

## Introduction

This project is focused on developing a hybrid quantum-classical neural network (QNN) for image classification tasks. The QNN is implemented using the `HybridQNN.py` module, which combines classical and quantum layers to create a more powerful neural network. The `Quanv2d.py` module is used to implement the quantum convolutional layer, which is a key component of the QNN.

The goal of this project is to explore the potential of QNNs for image classification tasks, and to compare their performance to classical neural networks. We hope that this project will contribute to the growing field of quantum machine learning, and inspire further research in this area.

## Model

The `Quanv2d.py` module is used to implement the quantum convolutional layer, which is a key component of the hybrid quantum-classical neural network (QNN) for image classification tasks. This module defines the Quanv2d class, which inherits from the PyTorch nn.Module class and implements the quantum convolutional layer using the qiskit library.

The `HybridQNN.py` module combines classical and quantum layers to create a more powerful neural network. It defines the HybridQNN class, which inherits from the PyTorch nn.Module class and implements the hybrid QNN using the Quanv2d class and other PyTorch layers. The HybridQNN class also includes methods for training and evaluating the QNN on image classification tasks.

### Model Summary

| Layer | Type | Input Shape | Output Shape |
|-------|------|-------------|--------------|
| `conv1` | Conv2d | (batch_size, 1, 28, 28) | (batch_size, 1, 26, 26) |
| `bn1` | BatchNorm2d | (batch_size, 1, 26, 26) | (batch_size, 1, 26, 26) |
| `sigmoid` | Sigmoid | (batch_size, 1, 26, 26) | (batch_size, 1, 26, 26) |
| `maxpool1` | MaxPool2d | (batch_size, 1, 26, 26) | (batch_size, 1, 13, 13) |
| `conv2` | Quanv2d | (batch_size, 1, 13, 13) | (batch_size, 2, 11, 11) |
| `bn2` | BatchNorm2d | (batch_size, 2, 11, 11) | (batch_size, 2, 11, 11) |
| `relu2` | ReLU | (batch_size, 2, 11, 11) | (batch_size, 2, 11, 11) |
| `maxpool2` | MaxPool2d | (batch_size, 2, 11, 11) | (batch_size, 2, 5, 5) |
| `flatten` | Flatten | (batch_size, 2, 5, 5) | (batch_size, 50) |
| `linear` | Linear | (batch_size, 50) | (batch_size, 10) |

## Extensibility

The Quanv2d class in `Quanv2d.py` is designed to be easily extensible, so that users can modify the quantum circuit to suit their needs. Users can inherit from the Quanv2d class and override the `build_circuit()` method to define their own quantum circuit. Here's an example of how to do this

```python
from Quanv2d import Quanv2d
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

class CustomQuanv2d(Quanv2d):
    def __init__(self,input_channel,output_channel,num_qubits,num_weight,kernel_size = 3,stride = 1):
        super().__init__(input_channel, output_channel, num_qubits, num_weight, kernel_size, stride)
    def build_circuit(self,num_weights : int,num_input : int,num_qubits : int = 3):
        qc = QuantumCircuit(num_qubits)
        weight_params = [Parameter('w{}'.format(i)) for i in range(num_weights)]
        input_params = [Parameter('x{}'.format(i)) for i in range(num_input)]
        '''
        Build your own quantum circuit here
        '''
        return qc, weight_params, input_params
```

## Prerequisites

Before running the code in this repository, you will need to set up your environment. Here are the steps to follow:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/miles0428/Hackthon-NCCU-2023.git
    ```

2. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

    This will install all the necessary packages, including PyTorch, qiskit, and matplotlib.

## License Information

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for more information.

## References

- [HHybrid Quantum-Classical Convolutional Neural Networks](https://arxiv.org/pdf/1911.02998.pdf)
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Github Copilot](https://thanksforthecode.com)
