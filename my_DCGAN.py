import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10
import numpy as np
from torch  import optim
import torchvision.utils as vutil

class Config:
    lr = 0.0002
    nz = 100  # 噪声维度
    image_size = 64
    image_size2 = 64
    nc = 3  # 图片三通道
    ngf = 64  # 生成图片
    ndf = 64  # 判别图片
    gpuids = None
    beta1 = 0.5
    batch_size = 32
    max_epoch = 1  # =1 when debug
    workers = 2
opt = Config()
# 数据加载和预处理
dataset = CIFAR10(root='cifar10/',train=True, download=True,
                  transform=transforms.Compose(
                      [transforms.Scale(opt.image_size),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5] * 3, [0.5] * 3)
                       ]))
# 什么惰性加载，预加载，多线程，乱序  全都解决
dataloader = t.utils.data.DataLoader(dataset, opt.batch_size,True, num_workers=opt.workers)

# 模型定义
class ModelG(nn.Module):
    def __init__(self, ngpu):
        super(ModelG, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential()
        self.model.add_module('deconv1', nn.ConvTranspose2d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False))
        self.model.add_module('bnorm1', nn.BatchNorm2d(opt.ngf * 8))
        self.model.add_module('relu1', nn.ReLU(True))

        self.model.add_module('deconv2', nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False))
        self.model.add_module('bnorm2', nn.BatchNorm2d(opt.ngf * 4))
        self.model.add_module('relu2', nn.ReLU(True))

        self.model.add_module('deconv3', nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False))
        self.model.add_module('bnorm3', nn.BatchNorm2d(opt.ngf * 2))
        self.model.add_module('relu3', nn.ReLU(True))

        self.model.add_module('deconv4', nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False))
        self.model.add_module('bnorm4', nn.BatchNorm2d(opt.ngf))
        self.model.add_module('relu4', nn.ReLU(True))

        self.model.add_module('deconv5', nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False))
        self.model.add_module('tanh', nn.Tanh())

    def forward(self, input):
        gpuids = None
        if self.ngpu:
            gpuids = range(gpuids)
        return nn.parallel.data_parallel(self.model, input, device_ids=gpuids)

def weight_init(m):
    # 模型参数初始化．　可以优化成为xavier 初始化
    class_name = m.__class__.__name__
    if class_name.find('conv') != -1:
        m.weight.data.normal_(0, 0.02)
    if class_name.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)

class ModelD(nn.Module):
    def __init__(self, ngpu):
        super(ModelD, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential()
        self.model.add_module('conv1', nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False))
        self.model.add_module('relu1', nn.LeakyReLU(0.2, inplace=True))

        self.model.add_module('conv2', nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False))
        self.model.add_module('bnorm2', nn.BatchNorm2d(opt.ndf * 2))
        self.model.add_module('relu2', nn.LeakyReLU(0.2, inplace=True))

        self.model.add_module('conv3', nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False))
        self.model.add_module('bnorm3', nn.BatchNorm2d(opt.ndf * 4))
        self.model.add_module('relu3', nn.LeakyReLU(0.2, inplace=True))

        self.model.add_module('conv4', nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False))
        self.model.add_module('bnorm4', nn.BatchNorm2d(opt.ndf * 8))
        self.model.add_module('relu4', nn.LeakyReLU(0.2, inplace=True))

        self.model.add_module('conv5', nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False))
        self.model.add_module('sigmoid', nn.Sigmoid())

    def forward(self, input):
        gpuids = None
        if self.ngpu:
            gpuids = range(gpuids)
        return nn.parallel.data_parallel(self.model, input, device_ids=gpuids).view(-1, 1)


netg = ModelG(opt.gpuids)
netg.apply(weight_init)         # 非常好的权重初始化方法
netd = ModelD(opt.gpuids)
netd.apply(weight_init)
# 优化器
optimizerD=optim.Adam(netd.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
optimizerG=optim.Adam(netg.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))

# 模型的输入输出
input=Variable(t.FloatTensor(opt.batch_size,opt.nc,opt.image_size,opt.image_size2))
label=Variable(t.FloatTensor(opt.batch_size))
noise=Variable(t.FloatTensor(opt.batch_size,opt.nz,1,1))
fixed_noise=Variable(t.FloatTensor(opt.batch_size,opt.nz,1,1).normal_(0,1))
real_label=1
fake_label=0

# 训练

criterion = nn.BCELoss()
for epoch in range(6):
    for ii, data in enumerate(dataloader, 0):
        # 训练　Ｄ　网
        netd.zero_grad()
        # 真实图片
        real, _ = data
        input.data.resize_(real.size()).copy_(real)
        label.data.resize_(input.size()[0]).fill_(real_label)
        output = netd(input)
        error_real = criterion(output, label)
        error_real.backward()
        D_x = output.data.mean()
        # 假图片
        noise.data.resize_(input.size()[0], opt.nz, 1, 1).normal_(0, 1)
        fake_pic = netg(noise).detach()
        output2 = netd(fake_pic)
        label.data.fill_(fake_label)
        error_fake = criterion(output2, label)
        error_fake.backward()
        D_x2 = output2.data.mean()
        error_D = error_real + error_fake
        optimizerD.step()

        # 训练 G网  G网和D网训练次数1:2
        if t.rand(1)[0] > 0.5:
            netg.zero_grad()
            label.data.fill_(real_label)
            noise.data.normal_(0, 1)
            fake_pic = netg(noise)
            output = netd(fake_pic)
            error_G = criterion(output, label)
            error_G.backward()
            optimizerG.step()
            D_G_z2 = output.data.mean()

        print('{ii}/{epoch}     lossD:{error_D},lossG:{error_G},{D_x2},{D_G_z2},{D_x}'.format(ii=ii, epoch=epoch, \
                                                                                              error_D=error_D.data[0],
                                                                                              error_G=error_G.data[0], \
                                                                                              D_x2=D_x2, D_G_z2=D_G_z2,
                                                                                              D_x=D_x))
        if ii % 100 == 0 and ii > 0:
            fake_u = netg(fixed_noise)
            vutil.save_image(fake_u.data, 'fake%s.png' % ii)
            vutil.save_image(real, 'real%s.png' % ii)
t.save(netd.state_dict(),'1epoch_netd.pth')
t.save(netg.state_dict(),'1epoch_netg.pth')