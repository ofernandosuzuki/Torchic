from abc import ABC, abstractmethod
from typing import List
import numpy as np
import math


class Activation(ABC):
        @abstractmethod
        def forward(self, z: np.ndarray) -> np.ndarray:                 
                pass
        
        @abstractmethod
        def backward(self, z: np.ndarray) -> np.ndarray:          
                pass
        
        def __call__(self, z: np.ndarray):
                return self.forward(z)


class Sigmoid(Activation):
        def forward(self, z: np.ndarray) -> np.ndarray:                        
                return 1 / (1 + np.exp(-z))
                
        def backward(self, z: np.ndarray) -> np.ndarray:                        
                a = self.forward(z)
                return a * (1 - a)
        
        def __str__(self) -> str:
                return 'Sigmoid'

class TanH(Activation):
        def forward(self, z: np.ndarray) -> np.ndarray:                     
                return np.tanh(z)
                
        def backward(self, z: np.ndarray) -> np.ndarray:                   
                return 1 - np.power(self.forward(z), 2)

        def __str__(self) -> str:
                return 'TanH'


class ReLU(Activation):
        def forward(self, z: np.ndarray) -> np.ndarray:                        
                return np.maximum(0, z)
                
        def backward(self, z: np.ndarray) -> np.ndarray:                        
                return np.where(z > 0, 1, 0)   
        
        def __str__(self) -> str:
                return 'ReLU'


class LeakyReLU(Activation):
        def __init__(self, alpha = 0.01):
                super().__init__()
                self.alpha = alpha
        
        def forward(self, z: np.ndarray) -> np.ndarray:                        
                return np.where(z > 0, z, self.alpha * z)
                
        def backward(self, z: np.ndarray) -> np.ndarray:                    
                return np.where(z > 0, 1, self.alpha)
        
        def __str__(self) -> str:
                return 'Leaky ReLU'
        
        
class Linear(Activation):
        def forward(self, z: np.ndarray) -> np.ndarray:                        
                return z
        
        def backward(self, z: np.ndarray) -> np.ndarray:                       
                return np.ones_like(z) 

        def __str__(self) -> str:
                return 'Linear'


class Softmax(Activation):
        def __init__(self):
                super().__init__()
                self.A = None
                
        def forward(self, z: np.ndarray):
                z_stable = z - np.max(z, axis=1, keepdims=True)
                z_stable = np.clip(z_stable, -100, 100)                               
                exp_z = np.exp(z_stable)                                      
                return  exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        def backward(self):
                pass
        
        def __str__(self) -> str:
                return 'Softmax'


#!----------------------------------------------------------------------------------------------------------------------
class Cost(ABC):
        def __init__(self):
                super().__init__()
                self.y_pred = None
                self.y_true = None
        
        def __call__(self, y_pred: np.ndarray, y_true: np.ndarray):
                return self.forward(y_pred, y_true)
        
        @abstractmethod
        def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
                pass
        
        @abstractmethod
        def backward(self):
                pass
        
        
class MeanSquareError(Cost):
        def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
                self.y_pred, self.y_true = y_pred, y_true
                return np.mean((y_pred - y_true)**2)
        
        def backward(self):
                m = self.y_pred.shape[0]
                return (self.y_pred - self.y_true) / m
        
        def __str__(self) -> str:
                return 'MSE'


class CrossEntropyLoss(Cost):
        def __init__(self):
                super().__init__()
                self.y_true_one_hot = None
                self.epsilon = 1e-15
        
        def forward(self, y_pred: np.ndarray, y_true: np.ndarray):    
                self.y_pred, self.y_true = y_pred, y_true
                samples, classes = self.y_pred.shape
                
                y_true_one_hot = np.zeros((samples, classes))
                y_true_one_hot[np.arange(samples), self.y_true] = 1
                self.y_true_one_hot = y_true_one_hot
                
                log_likelihood = np.sum(y_true_one_hot * np.log(self.y_pred + self.epsilon), axis=1)
                loss = - np.sum(log_likelihood) / samples
                return loss

        def backward(self):
                return (self.y_pred - self.y_true_one_hot) / self.y_pred.shape[0]
        
        def __str__(self) -> str:
                return 'Cross Entropy'
        
        
class BinaryCrossEntropyLoss(Cost):
        def forward(self):
                epsilon = 1e-15
                y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)
                loss = -(self.y_true * np.log(y_pred) + (1 - self.y_true) * np.log(1 - y_pred))
                return loss
        
        def backward(self):
                return self.y_pred - self.y_true
        
        def __str__(self) -> str:
                return 'Binary Cross Entropy'


#!----------------------------------------------------------------------------------------------------------------------
class Optimizer(ABC):
        def __init__(self, parameters: List, lr: float):
                super().__init__()
                self._parameters = parameters
                self._learning_rate = lr

        @abstractmethod
        def step(self):
                pass
        
        def zero_grad(self):
                for layer in self.parameters:
                        layer.theta_grad = np.zeros_like(layer.theta)
                        layer.bias_grad = np.zeros_like(layer.bias)
        
        @property
        def parameters(self):
                return self._parameters
                
        @property
        def learning_rate(self):
                return self._learning_rate
                
        @parameters.setter
        def parameters(self, layers: List):
                self._parameters = layers
        
        @learning_rate.setter
        def learning_rate(self, lr: float):
                self._learning_rate = lr
        
        
class SGD(Optimizer):
        def step(self):
                for layer in self.parameters:
                        layer.theta -= self.learning_rate * layer.theta_grad
                        layer.bias -= self.learning_rate * layer.bias_grad
                        
        def __str__(self) -> str:
                return f'SGD(learning_rate={self.learning_rate})'
                        
 
class Adam(Optimizer):
        def __init__(self, parameters: List, lr: float, weight_decay: float = None):
                super().__init__(parameters, lr)
                self.weight_decay = weight_decay
                self.epsilon = 1e-15
                self.beta = 0.9
                self.gamma = 0.999
                
                self.m_theta = {}
                self.v_theta = {} 
                self.m_bias = {}
                self.v_bias = {}
                self.t = 0
                 
                for i, layer in enumerate(self.parameters):
                        self.m_theta[i] = np.zeros_like(layer.theta)
                        self.v_theta[i] = np.zeros_like(layer.theta)                
                        self.m_bias[i] = np.zeros_like(layer.bias)
                        self.v_bias[i] = np.zeros_like(layer.bias)

        def step(self): 
                self.t += 1    
                for i,layer in enumerate(self.parameters):
                        dJ_dTheta, dJ_dB = layer.theta_grad, layer.bias_grad
                        
                        if self.weight_decay:
                                dJ_dTheta = dJ_dTheta + self.weight_decay * layer.theta
                                dJ_dB = dJ_dB + self.weight_decay * layer.bias

                        self.m_theta[i] = self.beta * self.m_theta[i] + (1 - self.beta) * dJ_dTheta
                        self.m_bias[i] = self.beta * self.m_bias[i] + (1 - self.beta) * dJ_dB

                        self.v_theta[i] = self.gamma * self.v_theta[i] + (1 - self.gamma) * (dJ_dTheta ** 2)
                        self.v_bias[i] = self.gamma * self.v_bias[i] + (1 - self.gamma) * (dJ_dB ** 2)

                        m_theta_corrected = self.m_theta[i] / (1 - self.beta ** self.t)
                        m_bias_corrected = self.m_bias[i] / (1 - self.beta ** self.t)

                        v_theta_corrected = self.v_theta[i] / (1 - self.gamma ** self.t)
                        v_bias_corrected = self.v_bias[i] / (1 - self.gamma ** self.t)

                        layer.theta -= self.learning_rate * m_theta_corrected / (np.sqrt(v_theta_corrected) + self.epsilon)
                        layer.bias -= self.learning_rate * m_bias_corrected  / (np.sqrt(v_bias_corrected) + self.epsilon)

        def __str__(self) -> str:
                return f'Adam(learning_rate={self.learning_rate}, beta={self.beta}, gamma={self.gamma}, epsilon={self.epsilon}, weight_decay={self.weight_decay})'

     
#!----------------------------------------------------------------------------------------------------------------------
def get_fan(parameters: np.ndarray, fan_mode: str) -> int:
        output_size, input_size = parameters.shape  
        if fan_mode == 'in':
                return input_size
        else:
                return output_size
        
def get_gain(activation: Activation) -> float:
        if isinstance(activation, (Sigmoid, Linear, Softmax)) :
                return 1.0
        if isinstance(activation, TanH):
                return 5.0/3.0
        if isinstance(activation, ReLU):
                return math.sqrt(2)
        if isinstance(activation, LeakyReLU):
                return math.sqrt(2/(1 + activation.alpha ** 2))
        return 1.0
        
        
#? https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
def xavier_uniform_initialization(parameters: np.ndarray, gain: float) -> np.ndarray:
        output_size, input_size = parameters.shape        
        std = gain * math.sqrt(2.0 / (input_size + output_size))
        a = math.sqrt(3.0) * std
        return np.random.uniform(low=-a, high=a, size=(output_size, input_size))


#? https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_
def xavier_normal_initialization(parameters: np.ndarray, gain: float) -> np.ndarray:
        output_size, input_size = parameters.shape        
        std = gain * math.sqrt(2.0 / (input_size + output_size))
        return np.random.normal(loc=0.0, scale=std, size=(output_size, input_size))


#? https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
def kaiming_uniform_initialization(parameters: np.ndarray, gain: float, fan_mode: str):
        output_size, input_size = parameters.shape
        fan = get_fan(parameters, fan_mode)
        bound = gain * math.sqrt(3 / fan)
        return np.random.uniform(low=-bound, high=bound, size=(output_size, input_size))


#? https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
def kaiming_normal_initialization(parameters: np.ndarray, gain: float, fan_mode: str):
        output_size,input_size = parameters.shape
        fan = get_fan(parameters, fan_mode)
        std = gain / math.sqrt(fan)
        return np.random.normal(loc=0.0, scale=std, size=(output_size, input_size))
 

def explosion_initialization(parameters: np.ndarray):
        output_size, input_size = parameters.shape
        std = 100
        return np.random.normal(loc=0.0, scale=std, size=(output_size, input_size))


#!----------------------------------------------------------------------------------------------------------------------
class Layer:
        def __init__(self, input_size: int, output_size: int, activation: Activation, initialization: str) -> None:
                self.input_size = input_size
                self.output_size = output_size
                self.activation = activation
                self.initialization = initialization
                self.fan_mode = 'in'
                self.theta, self.bias = self.__initialization_parameters()
                
                self.input = None
                self.z = None
                self.a = None
                
                self.theta_grad = None
                self.bias_grad = None
        
        def __call__(self, input: np.ndarray) -> np.ndarray:
                return self.forward(input)
        
        def forward(self, input: np.ndarray) -> np.ndarray:
                self.input = input
                self.z = np.dot(input, self.theta.T) + self.bias
                self.a = self.activation(self.z)
                return self.a
            
        def backward(self, dJ_dA: np.ndarray) -> np.ndarray:
                if isinstance(self.activation, (Softmax, Linear)):
                        delta = dJ_dA
                else:
                        delta = dJ_dA * self.activation.backward(self.z)                           
                
                self.theta_grad = np.dot(delta.T, self.input)                           
                self.bias_grad = np.sum(delta, axis=0, keepdims=True)                  
        
                deltaL_thetaL = np.dot(delta, self.theta)                               
                return deltaL_thetaL                                               
        
        def __initialization_parameters(self) -> np.ndarray:
                theta = np.zeros((self.output_size, self.input_size))                   
                bias = np.zeros((1, self.output_size))       
                gain = get_gain(self.activation)
                
                # explosivo
                if self.initialization == 'explosive':
                        theta = explosion_initialization(theta)
                
                # Xavier
                if isinstance(self.activation, (Linear, Sigmoid, TanH, Softmax)):
                        if self.initialization == 'uniform':
                                theta = xavier_uniform_initialization(theta, gain)
                        elif self.initialization == 'normal':
                                theta = xavier_normal_initialization(theta, gain)

                # kaiming
                if isinstance(self.activation, (ReLU, LeakyReLU)):
                        if self.initialization == 'uniform':
                                theta = kaiming_uniform_initialization(theta, gain, self.fan_mode)
                        elif self.initialization == 'normal':
                                theta = kaiming_normal_initialization(theta, gain, self.fan_mode)
                return theta, bias

                
#!----------------------------------------------------------------------------------------------------------------------
class Dataloader:
        def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int):
                self.X = X
                self.y = y
                self.batch_size = batch_size
                self.num_samples = X.shape[0]
                
        def __iter__(self):
                for i in range(0, self.num_samples, self.batch_size):
                        X_batch = self.X[i : i+self.batch_size]
                        y_batch = self.y[i : i+self.batch_size]
                        batch_index = i // self.batch_size
                        yield batch_index, (X_batch, y_batch)
                
        def __len__(self):
                return int(np.ceil(self.num_samples / self.batch_size))
