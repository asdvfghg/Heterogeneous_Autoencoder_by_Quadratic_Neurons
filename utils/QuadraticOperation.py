import math
import torch
import torch.nn as nn
from torch.nn import Parameter, init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

class QuadraticOperation(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(QuadraticOperation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = Parameter(torch.Tensor(in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(in_features, out_features))
        self.weight_b = Parameter(torch.Tensor(in_features, out_features))



        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        self.__reset_bias()

    def forward(self, x):
        if use_gpu:
            self.weight_r.cuda()
            self.weight_g.cuda()
            self.weight_b.cuda()
            self.bias_r.cuda()
            self.bias_g.cuda()
            self.bias_b.cuda()
        out = (torch.matmul(x, self.weight_r) + self.bias_r)*(torch.matmul(x, self.weight_g) + self.bias_g)\
              + torch.matmul(torch.pow(x, 2), self.weight_b) + self.bias_b
        return out

if __name__ == '__main__':
    a = torch.randn(10, 28*28)
    b = QuadraticOperation(28*28, 128)
    c = b(a)
    print(c.shape)