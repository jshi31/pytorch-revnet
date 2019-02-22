import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import pdb

def SVD_op(param, Lambda):
    """ get SVD_ops
    :param param: param (bs, n, n)
    :param param: Lambda(bs, n)
    :return: W (bs, n, n)
    :return: W_inv (bs, n, n)
    """
    _, _, n = param.size()
    Q = torch.eye(n).expand_as(param).cuda()
    for i in range(30):
        v = param.index_select(2, torch.tensor([i]).cuda())
        v_T = v.transpose(1, 2)
        Q = Q - (2/(v_T@v)*Q)@v@v_T
    W = (Q * Lambda.unsqueeze(1)) @ Q.transpose(1, 2)
    W_inv = Q * (1/Lambda.unsqueeze(1)) @ Q.transpose(1, 2)
    return W, W_inv

class GroupLinear(nn.Module):
    def __init__(self, group, channels):
        super(GroupLinear, self).__init__()
        self.group = group
        self.channels = channels
        self.weight = nn.Parameter(torch.Tensor(group, channels, channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def reverse(self):
        self.rev_weight = torch.inverse(self.weight.data)
        return self.rev_weight

    def apply_rev(self, y):
        pdb.set
        rev_weight = torch.inverse(self.weight.data)
        x = y @ rev_weight
        return x

    def forward(self, x):
        """
        :param x: (group, bs, channels)
        :return:
        """
        x = x @ self.weight
        return x


class OriOrthLinear(nn.Module):
    """ orthogonal matrix linear layer with activation stored
    """
    def __init__(self, group, channels):
        super(OriOrthLinear, self).__init__()
        self.group = group
        self.channels = channels
        self.weight = nn.Parameter(torch.Tensor(group, channels, channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: (group, bs, channels)
        :return:
        """
        # check device
        cuda_check = x.is_cuda
        if cuda_check:
            device = x.get_device()
        # x (group, bs, channel)
        for i in range(self.channels):
            weight = self.weight.index_select(-1, torch.tensor([i], dtype=torch.long, device=device)).squeeze(-1)
            x = self.forward_(x, weight)
        return x

    def forward_(self, x, weight):
        """
        :param x: (group, bs, channels)
        :param weight: (group, channels)
        :return:
        """
        # weight(bs, n, 1)
        weight = weight.unsqueeze(-1)
        weight_T = weight.transpose(1, 2)
        output = x - 2/(weight_T@weight)*x@weight@weight_T
        return output




class OrthFunction(Function):
    """a single step to compose orthogonal matrix"""
    @staticmethod
    def forward(ctx, input, weight, activations=[]):
        """
        :param ctx:
        :param input: (group, m, n)
        :param weight: (group, n)
        :return: output: (group, m, n)
        """
        # weight(bs, n, 1)
        ctx.save_for_backward(weight)
        with torch.no_grad():
            weight = weight.unsqueeze(-1)
            weight_T = weight.transpose(1, 2)
            output = input - 2/(weight_T@weight)*input@weight@weight_T
        ctx.activations = activations
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: (group, m, n)
        :return: (grad_input, grad_weight)
        """
        weight, = ctx.saved_tensors
        output = ctx.activations.pop()
        # grad_output requires grad
        group, _, n = grad_output.shape
        # weight(group, n, 1)
        weight = weight.clone().data.unsqueeze(-1)
        weight_T = weight.transpose(1, 2)
        # initialize gradient to None
        grad_input = grad_weight = None
        G = weight_T @ weight
        input = output - (2/G*output) @ weight @ weight_T
        ctx.activations.append(input)
        input_T = input.transpose(1, 2)
        tmp = input_T @ grad_output / G
        # tmp = torch.ones(group, n, n, device='cuda')
        if ctx.needs_input_grad[0]:
            grad_input = grad_output - (2/G*grad_output) @ weight @ weight_T
        if ctx.needs_input_grad[1]:
            grad_weight = -2 * ((tmp * weight_T).sum(2) + (tmp * weight).sum(1) - \
                               ((tmp * (weight @ weight_T)).sum(1, keepdim=True).sum(2, keepdim=True)/(G) * 2 * weight).squeeze(-1))
        return grad_input, grad_weight, None

class OrthBIFunction(Function):
    """a helper function to non-activation orthogonal matrix using build in autograd"""
    @staticmethod
    def forward(ctx, input, weight, activations=[]):
        """
        :param ctx:
        :param input: (group, m, n)
        :param weight: (group, n)
        :return: output: (group, m, n)
        """
        # weight(bs, n, 1)
        ctx.save_for_backward(weight)
        with torch.no_grad():
            weight = weight.unsqueeze(-1)
            weight_T = weight.transpose(1, 2)
            output = input - 2/(weight_T@weight)*input@weight@weight_T
        ctx.activations = activations
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: (group, m, n)
        :return: (grad_input, grad_weight)
        """
        weight, = ctx.saved_tensors
        output = ctx.activations.pop()
        # grad_output requires grad
        group, _, n = grad_output.shape
        # weight(group, n, 1)
        weight = weight.clone().data.unsqueeze(-1)
        weight_T = weight.transpose(1, 2)
        G = weight_T @ weight
        input = output - (2/G*output) @ weight @ weight_T
        ctx.activations.append(input)
        grad_input, grad_weight = _grad(input, weight, grad_output)
        return grad_input, grad_weight.squeeze(-1), None


def _grad(x, weight, grad_output):
    x.requires_grad()
    weight.requires_grad()
    with torch.enable_grad:
        weight_T = weight.transpose(1, 2)
        G = weight_T @ weight
        # forward again
        y = x - 2/G*x@weight@weight_T
        dx, dw = torch.autograd.grad(y, (x, weight), grad_output)
    return dx, dw


class OrthLinear(nn.Module):
    """ orthogonal matrix without storing activation.
    """
    def __init__(self, group, channels, activations=[]):
        super(OrthLinear, self).__init__()
        self.group = group
        self.channels = channels
        self.activations = activations
        self.weight = nn.Parameter(torch.Tensor(group, channels, channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: (group, bs, in_channel)
        :return:
        """
        # check device
        cuda_check = x.is_cuda
        if cuda_check:
            device = x.get_device()
        # x (group, bs, channel)
        for i in range(self.channels):
            weight = self.weight.index_select(-1, torch.tensor([i], dtype=torch.long, device=device)).squeeze(-1)
            x = OrthFunction.apply(x, weight, self.activations)
        self.activations.append(x.data)
        return x

    def free(self):
        del self.activations[:]


class OrthBILinear(nn.Module):
    """ orthogonal matrix without storing activation.
    """
    def __init__(self, group, channels, activations=[]):
        super(OrthBILinear, self).__init__()
        self.group = group
        self.channels = channels
        self.activations = activations
        self.weight = nn.Parameter(torch.Tensor(group, channels, channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: (group, bs, in_channel)
        :return:
        """
        # check device
        cuda_check = x.is_cuda
        if cuda_check:
            device = x.get_device()
        # x (group, bs, channel)
        for i in range(self.channels):
            weight = self.weight.index_select(-1, torch.tensor([i], dtype=torch.long, device=device)).squeeze(-1)
            x = OrthBIFunction.apply(x, weight, self.activations)
        self.activations.append(x.data)
        return x

    def free(self):
        del self.activations[:]


