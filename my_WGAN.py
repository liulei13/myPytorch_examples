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
#from tensorboard_logger import Logger

'''
https://zhuanlan.zhihu.com/p/25071913
WGAN 相比于DCGAN 的修改：
1. 判别器最后一层去掉sigmoid                                       # 回归问题,而不是二分类概率
2. 生成器和判别器的loss不取log                                      # Wasserstein 距离
3. 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c          #Ｗ距离－＞Ｌ连续－＞数值稳定
4. 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行　 ＃－＞玄学

GAN 两大问题的解释：
collapse mode ->KL 散度不对称
数值不稳定 -> KL散度和JS散度优化方向不一样
'''


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
    max_epoch = 12  # =1 when debug
    workers = 2
    clamp_num = 0.01  # WGAN 截断大小
opt = Config()

# 加载数据
dataset=CIFAR10(root='cifar10/',download=True,
                transform=transforms.Compose(\
                                             [transforms.Scale(opt.image_size) ,
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5]*3,[0.5]*3)
                                             ]))
# 什么惰性加载，预加载，多线程，乱序  全都解决
dataloader=t.utils.data.DataLoader(dataset,opt.batch_size,True,num_workers=opt.workers)


# 网络结构

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
    # 参数初始化。 可以改成xavier初始化方法
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
        # modify： remove sigmoid
        # self.model.add_module('sigmoid',nn.Sigmoid())

    def forward(self, input):
        gpuids = None
        if self.ngpu:
            gpuids = range(gpuids)
        return nn.parallel.data_parallel(self.model, input, device_ids=gpuids).view(-1, 1).mean(0).view(1)  #
        ## no loss but score


netg = ModelG(opt.gpuids)
netg.apply(weight_init)
netd = ModelD(opt.gpuids)
netd.apply(weight_init)
# 定义优化器
optimizerD=optim.RMSprop(netd.parameters(),lr=opt.lr ) #modify ： 不要采用基于动量的优化方法 如Adam
optimizerG=optim.RMSprop(netg.parameters(),lr=opt.lr )  #

# 定义 D网和G网的输入
input=Variable(t.FloatTensor(opt.batch_size,opt.nc,opt.image_size,opt.image_size2))
label=Variable(t.FloatTensor(opt.batch_size))
noise=Variable(t.FloatTensor(opt.batch_size,opt.nz,1,1))
fixed_noise=Variable(t.FloatTensor(opt.batch_size,opt.nz,1,1).normal_(0,1))
real_label=1
fake_label=0
# criterion=nn.BCELoss() # WGAN 不需要log（交叉熵）
one = t.FloatTensor([1])
mone = -1 * one

# 开始训练
for epoch in range(opt.max_epoch):
    for ii, data in enumerate(dataloader, 0):
        #### 训练D网 ####
        netd.zero_grad() # 有必要
        real, _ = data
        input.data.resize_(real.size()).copy_(real)
        label.data.resize_(input.size()[0]).fill_(real_label)
        output = netd(input)
        output.backward(one)  #######for wgan
        D_x = output.data.mean()

        noise.data.resize_(input.size()[0], opt.nz, 1, 1).normal_(0, 1)
        fake_pic = netg(noise).detach()
        output2 = netd(fake_pic)
        label.data.fill_(fake_label)
        output2.backward(mone)  # for wgan
        D_x2 = output2.data.mean()
        optimizerD.step()
        for parm in netd.parameters(): parm.data.clamp_(-opt.clamp_num, opt.clamp_num)  ### 只有判别器需要 截断参数

        #### 训练G网 ########
        if t.rand(1)[0] > 0.8:
            # d网和g网的训练次数不一样, 这里d网和g网的训练比例大概是: 5:1
            netg.zero_grad()
            label.data.fill_(real_label)
            noise.data.normal_(0, 1)
            fake_pic = netg(noise)
            output = netd(fake_pic)
            output.backward(one)
            optimizerG.step()
            # for parm in netg.parameters():parm.data.clamp_(-opt.clamp_num,opt.clamp_num)## 只有判别器需要 生成器不需要
            D_G_z2 = output.data.mean()

        if ii % 100 == 0 and ii > 0:
            fake_u = netg(fixed_noise)
            vutil.save_image(fake_u.data, 'wgan/fake%s_%s.png' % (epoch, ii))
            vutil.save_image(real, 'wgan/real%s_%s.png' % (epoch, ii))

t.save(netd.state_dict(),'epoch_netd.pth')
t.save(netg.state_dict(),'epoch_netg.pth')