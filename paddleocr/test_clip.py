import paddle
import numpy as np 
import torch

from paddle.nn.initializer import KaimingNormal, Constant

zeros_ = Constant(value=0.)
ones_ = Constant(value=40.)


x = np.random.rand(4, 3, 256)
x_pd = paddle.to_tensor(x, dtype='float32')
x_th = torch.Tensor(x)

# paddle 
linear_pd = paddle.nn.Linear(in_features=256, out_features=10,
                          weight_attr=paddle.ParamAttr(need_clip=True, initializer=ones_),
                          bias_attr=False)


out_pd = linear_pd(x_pd)
loss_pd = paddle.mean(out_pd)
loss_pd.backward()

clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=20.0)
sdg_pd = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear_pd.parameters(), grad_clip=clip)
sdg_pd.step()

# torch
linear_th = torch.nn.Linear(256, 10)
for m in linear_th.modules():
    m.weight.data.fill_(40)
    m.bias.data.zero_()


out_th = linear_th(x_th)
loss_th = torch.mean(out_th)
loss_th.backward()
torch.nn.utils.clip_grad_norm_(linear_th.parameters(), 20, 2)
sgd_th = torch.optim.SGD(linear_th.parameters(), lr=0.1)
sgd_th.step()
import pdb 
pdb.set_trace()
print()

