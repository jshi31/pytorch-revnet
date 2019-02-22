import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from orthmat import OrthFunction, OrthLinear, OriOrthLinear, OrthBILinear, GroupLinear
import time
import pdb

def test_grad():
    input = (torch.randn(2, 30, 20, requires_grad=True).double(), torch.randn(2, 20, requires_grad=True).double())
    test = gradcheck(OrthFunction.apply, input, eps=1e-6, atol=1e-4)
    print(test)

def testorth(group, bs, n):
    fc = OrthLinear(group, n)
    fc.to('cuda')
    optimizer = torch.optim.Adam(fc.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    lo = 0
    t0 = time.time()
    t1 = time.time()
    tf = 0
    tb = 0
    t = 0
    for i in range(50000):
        input = torch.randn(group, bs, n, requires_grad=True, device='cuda')
        output = fc(input)
        tf += (time.time() - t0)
        t0 = time.time()
        loss = criterion(output, input.data)
        optimizer.zero_grad()
        loss.backward()
        tb += (time.time() - t0)
        t0 = time.time()
        fc.free()
        optimizer.step()
        if i%500 + 1 == 0:
            pdb.set_trace()
        lo += loss.item()
        t += (time.time() - t1)
        t1 = time.time()
        print('epoch: {}, loss: {}, forward: {:0.3f}, backward: {:0.3f} time: {:0.3f}'.format(i, lo/(i + 1), tf/(i + 1),
                                                                                              tb/(i + 1), t/(i + 1)))

def testorthbi(group, bs, n):
    fc = OrthBILinear(group, n)
    fc.to('cuda')
    optimizer = torch.optim.Adam(fc.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    lo = 0
    t0 = time.time()
    t1 = time.time()
    tf = 0
    tb = 0
    t = 0
    for i in range(50000):
        input = torch.randn(group, bs, n, requires_grad=True, device='cuda')
        output = fc(input)
        tf += (time.time() - t0)
        t0 = time.time()
        loss = criterion(output, input.data)
        optimizer.zero_grad()
        loss.backward()
        tb += (time.time() - t0)
        t0 = time.time()
        fc.free()
        optimizer.step()
        if i%500 + 1 == 0:
            pdb.set_trace()
        lo += loss.item()
        t += (time.time() - t1)
        t1 = time.time()
        print('epoch: {}, loss: {}, forward: {:0.3f}, backward: {:0.3f} time: {:0.3f}'.format(i, lo/(i + 1), tf/(i + 1),
                                                                                              tb/(i + 1), t/(i + 1)))
def testoriorth(group, bs, n):
    fc = OriOrthLinear(group, n)
    fc.to('cuda')
    optimizer = torch.optim.Adam(fc.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    lo = 0
    t0 = time.time()
    t1 = time.time()
    tf = 0
    tb = 0
    t = 0
    for i in range(50000):
        input = torch.randn(group, bs, n, requires_grad=True, device='cuda')

        output = fc(input)
        tf += (time.time() - t0)
        t0 = time.time()
        loss = criterion(output, input.data)
        optimizer.zero_grad()
        loss.backward()
        tb += (time.time() - t0)
        t0 = time.time()
        optimizer.step()
        if i%500 + 1 == 0:
            pdb.set_trace()
        lo += loss.item()
        t += (time.time() - t1)
        t1 = time.time()
        print('epoch: {}, loss: {}, forward: {:0.3f}, backward: {:0.3f} time: {:0.3f}'.format(i, lo/(i + 1), tf/(i + 1),
                                                                                              tb/(i + 1), t/(i + 1)))


def test(group, bs, n):
    fc = GroupLinear(group, n).to('cuda')
    optimizer = torch.optim.Adam(fc.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    lo = 0
    t0 = time.time()
    t1 = time.time()
    tf = 0
    tb = 0
    t = 0
    for i in range(50000):
        input = torch.rand(group, bs, n, requires_grad=True, device='cuda')
        output = fc(input)
        tf += (time.time() - t0)
        t0 = time.time()
        loss = criterion(output, input.data) # + 100*fc.weight.norm()
        if i % 10000 == -1:
            pdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        tb += (time.time() - t0)
        t0 = time.time()
        optimizer.step()
        lo += loss.item()
        t += (time.time() - t1)
        t1 = time.time()
        print('epoch: {}, loss: {}, forward: {:0.3f}, backward: {:0.3f} time: {:0.3f}'.format(i, lo/(i + 1), tf/(i + 1),
                                                                                              tb/(i + 1), t/(i + 1)))

def testinv(group, bs, n):
    fc = GroupLinear(group, n).to('cuda')
    optimizer = torch.optim.Adam(fc.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    lo = 0
    t0 = time.time()
    t1 = time.time()
    tf = 0
    tb = 0
    ti = 0
    t = 0
    for i in range(50000):
        input = torch.rand(group, bs, n, requires_grad=True, device='cuda')
        t0 = time.time()
        output = fc(input)
        tf += (time.time() - t0)
        t0 = time.time()

        loss = criterion(output, input.data) # + 100*fc.weight.norm()
        if i % 10000 == -1:
            pdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        tb += (time.time() - t0)
        optimizer.step()
        # inverse
        t0 = time.time()
        inp = fc.apply_rev(output)
        ti += (time.time() - t0)
        lo += loss.item()
        t += (time.time() - t1)
        t1 = time.time()
        print('epoch: {}, loss: {}, forward: {:0.3f}, backward: {:0.3f}, inverse: {:0.3f}, time: {:0.3f}'.format(i,
                                                                                                                 lo/(i + 1), tf/(i + 1),
                                                                                                                 tb/(i + 1), ti/(i + 1),
                                                                                                                 t/(i + 1)))

if __name__ == '__main__':
    group = 256
    bs = 16
    n = 512
    # test_grad()

    # testorth(group, bs, n)
    # testorthbi(group, bs, n)
    # testoriorth(group, bs, n)
    # test(group, bs, n)
    testinv(group, bs, n)
