import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

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


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # input，weight 都已经变成了 Tensor
        # 用 ctx 把该存的存起来，留着 backward 的时候用
        ctx.save_for_backward(input, weight, bias)
        output = input@(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # 由于 forward 只有一个 返回值，所以 backward 只需要一个参数 接收 梯度。
    # Since forward has only one return value, backward only needs one parameter to receive the gradient
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is Variable
        # unpack ten saved tensor (requires_grad=True)
        input, weight, bias = ctx.saved_variables
        # initialize the gradients with None
        grad_input = grad_weight = grad_bias = None

        # exam the need of gradient by: needs_input_grad
        # the number of return should be the same with the input.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t() @ input
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        # the order of gradient should be consistent with the order of the argument in forward
        return grad_input, grad_weight, grad_bias
