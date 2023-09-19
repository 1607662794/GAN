# 使用一个网上的例子试一下
import argparse
import os

import ignite
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import linear_sum_assignment
from skimage.metrics import peak_signal_noise_ratio as psnr

import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import torch_fidelity

parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器 https://docs.python.org/zh-cn/3/howto/argparse.html
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")  # 迭代次数
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")  # batch大小
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # 学习率
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")  # 动量梯度下降第一个参数
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")  # 动量梯度下降第二个参数
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")  # CPU个数
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")  # 噪声数据生成维度
parser.add_argument("--channels", type=int, default=3, help="number of image channels")  # 输入数据的通道数
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")  # 保存图像的迭代数
# parser.add_argument('--noise_type', type=str, default='normal')
parser.add_argument("--save_path", type=str, default=r'./logs',
                    help="Address where training parameters are stored ")  # 保存训练参数的地址
parser.add_argument("--real_images_folder", type=str, default=r'data/Plaster_side', help="real_images_folder ")  # 真实图片
parser.add_argument("--generated_images_folder", type=str, default=r'data/generated_images',
                    help="generated_images_folder")  # 生成图片
parser.add_argument("--resume", default=r'data/generated_images', help="checkpoint")  # 是否进行断点运算
parser.add_argument("--eraly_train", default=False, help="The switch to stop training early")  # 早停训练的开启
# parser.add_argument('--leading_metric', type=str, default='ISC', choices=('ISC', 'FID', 'KID', 'PPL' ))
parser.add_argument('--num_samples_for_metrics', type=int, default=16)
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")  # 输入数据的维度

opt = parser.parse_args()
print(opt)
print(*map(lambda m: ": ".join((m.__name__, m.__version__)), (torch, torchvision, ignite)), sep="\n")

cuda = True if torch.cuda.is_available() else False  # 判断GPU可用，有GPU用GPU，没有用CPU


def weights_init_normal(m):  # 自定义初始化参数
    classname = m.__class__.__name__  # 获得类名
    if classname.find("Conv") != -1:  # 在类classname中检索到了Conv
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim, 128 * (800 // 4) * (512 // 4)))  # l1函数进行Linear变换。线性变换的两个参数是变换前的维度，和变换之后的维度

        self.conv_blocks = nn.Sequential(  # nn.sequential{}是一个组成模型的壳子，用来容纳不同的操作
            nn.BatchNorm2d(128),  # BatchNorm2d的目的是使我们的一批（batch）feature map 满足均值0方差1，就是改变数据的量纲
            nn.Upsample(scale_factor=2),  # 上采样，将图片放大两倍（这就是为啥class最先开始将图片的长宽除了4，下面还有一次放大2倍）
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # 二维卷积函数，（输入数据channel，输出的channel，步长，卷积核大小，padding的大小）
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),  # relu激活函数
            nn.Upsample(scale_factor=2),  # 上采样
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # 二维卷积
            nn.BatchNorm2d(64, 0.8),  # BN
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),  # Tanh激活函数
        )

    def forward(self, z):
        out = self.l1(z)  # l1函数进行的是Linear变换 （第50行定义了）
        out = out.view(out.shape[0], 128, 800 // 4,
                       512 // 4)  # view是维度变换函数，可以看到out数据变成了四维数据，第一个是batch_size(通过整个的代码，可明白),第二个是channel，第三,四是单张图片的长宽
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]  # Conv卷积，Relu激活，Dropout将部分神经元失活，进而防止过拟合
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))  # 如果bn这个参数为True，那么就需要在block块里面添加上BatchNorm的归一化函数
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * (800 // 2 ** 4) * (512 // 2 ** 4), 1),
                                       nn.Sigmoid())  # 先进行线性变换，再进行激活函数激活
        # 上一句中 128是指model中最后一个判别模块的最后一个参数决定的，ds_size由model模块对单张图片的卷积效果决定的，而2次方是整个模型是选取的长宽一致的图片

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)  # 将处理之后的数据维度变成batch * N的维度形式
        validity = self.adv_layer(out)  # 第92行定义

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()  # 定义了一个BCE损失函数

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:  # 初始化，将数据放在cuda上
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# 数据预处理
transform = transforms.Compose([
    transforms.CenterCrop((800, 512)),  # 调整图片大小，这样调整的原因是后边总是涉及到了缩放，不然缩放不了
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])


# 数据加载和预处理

class CustomDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.image_files = [f for f in os.listdir(data_root) if f.lower().endswith(".tif")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_root, self.image_files[idx])

        try:
            image = Image.open(img_name).convert("RGB")
            # image = Image.open(img_name).convert("L")

            if self.transform:
                image = self.transform(image)

            # 返回数据（不返回标签）
            return image
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # 返回一个空tensor，表示数据加载失败
            return torch.empty((3, 64, 64))


# 加载自定义数据集
custom_dataset = CustomDataset(data_root="E:/Code/GAN/data/Plaster_side", transform=transform)
# DataLoader用于批量处理
dataloader = DataLoader(custom_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)

# # Configure data loader
# os.makedirs(r"E:\Code\GAN\data\Plaster_side", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(     #显卡加速
#     datasets.MNIST(
#         "../../data/mnist",                  #进行训练集下载
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# 加载预训练的Inception-v3模型
# inception_model = torchvision.models.inception_v3(pretrained=True)

# Optimizers                             定义神经网络的优化器  Adam就是一种优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr*2, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr,
                               betas=(opt.b1, opt.b2))  # 因为判别器太强，压制住了生成器，所以我降低判别器的学习率看看效果

# #动态更新学习率
# scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lambda step: 1. - step / opt.n_epochs*(108 // opt.batch_size))#学习率逐渐减小，最后减小至0
# scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lambda step: 1. - step / opt.n_epochs*(108 // opt.batch_size))

# 因为adam已经实现动态更新学习率了，所以现在不使用学习率调度器来进行处理
# scheduler_G = OneCycleLR(optimizer_G,
#                          max_lr=opt.lr * 2,  # Upper learning rate boundaries in the cycle for each parameter group
#                          steps_per_epoch= 108 // opt.batch_size,  # The number of steps per epoch to train for.
#                          epochs=opt.n_epochs,  # The number of epochs to train for.
#                          # total_steps= opt.n_epochs * (108 // opt.batch_size),
#                          # three_phase=True,
#                          anneal_strategy='cos')  # Specifies the annealing strategy
# scheduler_D = OneCycleLR(optimizer_D,
#                          max_lr=opt.lr * 2,  # Upper learning rate boundaries in the cycle for each parameter group
#                          steps_per_epoch= 108 // opt.batch_size,  # The number of steps per epoch to train for.
#                          epochs=opt.n_epochs,  # The number of epochs to train for.
#                          # total_steps= opt.n_epochs * (108 // opt.batch_size),
#                          # three_phase=True,
#                          anneal_strategy='cos')  # Specifies the annealing strategy

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# leading_metric, last_best_metric, metric_greater_cmp = {
#     'ISC': (torch_fidelity.KEY_METRIC_ISC_MEAN, 0.0, float.__gt__),
#     'FID': (torch_fidelity.KEY_METRIC_FID, float('inf'), float.__lt__),
#     'KID': (torch_fidelity.KEY_METRIC_KID_MEAN, float('inf'), float.__lt__),
#     'PPL': (torch_fidelity.KEY_METRIC_PPL_MEAN, float('inf'), float.__lt__),
#
# }[opt.leading_metric]

# ----------
#  Training
# ----------


# 创建SummaryWriter来记录损失到TensorBoard
writer = SummaryWriter(log_dir=f"runs")

if not os.path.exists('data/generated_images'):
    os.mkdir('data/generated_images')

learning_rates_G = []  # 用于可视化学习率变化
learning_rates_D = []  # 用于可视化学习率变化

# 进行断点重启运行
if opt.resume:
    if os.path.isfile("{}/checkpoint_model.pth".format(opt.save_path)):
        print("Resume from checkpoint...")
        checkpoint = torch.load("{}/checkpoint_model.pth".format(opt.save_path))
        generator.load_state_dict(checkpoint['model_state_dict_G'])
        discriminator.load_state_dict(checkpoint['model_state_dict_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_state_dict_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_state_dict_D'])
        # scheduler_G.load_state_dict(checkpoint['scheduler_state_dict_G'])
        # scheduler_D.load_state_dict(checkpoint['scheduler_state_dict_D'])
        initepoch = checkpoint['epoch'] + 1
        initstep = checkpoint['step'] + 1
        print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
    else:
        print("====>no checkpoint found.")
        initepoch = 0  # 如果没进行训练过，初始训练epoch值为1
        initstep = 0  # 如果没进行训练过，初始训练step值为1

step = initstep
for epoch in range(initepoch, opt.n_epochs):

    ssim_scores = []
    psnr_scores = []
    ssim_scores_trail = []  # 之前计算出来的数值有点问题，我新建一个列表，用来存放我自己的想法
    psnr_scores_trail = []  # 之前计算出来的数值有点问题，我新建一个列表，用来存放我自己的想法

    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)  # 会生成一个batch_size大小的一维列表
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))  # 将真实的图片转化为神经网络可以处理的变量

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()  # 把梯度置零  每次训练都将上一次的梯度置零，避免上一次的干扰

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (
            imgs.shape[0], opt.latent_dim))))  # 生成的噪音 随机构00维向量 均值0方差1维度(64，100)的噪音，随机初始化一个64大小batch的向量
        # 输入0到1之间，形状为imgs.shape[0], opt.latent_dim的随机高斯数据。np.random.normal()正态分布
        # Generate a batch of images
        gen_imgs = generator(z)  # 得到一个批次的图片

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()  # 反向传播和模型更新
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)  # 判别器判别真实图片是真的的损失
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)  # 判别器判别假图片是假的的损失
        d_loss = (real_loss + fake_loss) / 2  # 判别器去判别真实图片是真的和生成图片是假的的损失之和，让这个和越大，说明判别器越准确

        d_loss.backward()
        optimizer_D.step()

        learning_rates_G.append(optimizer_G.param_groups[0]["lr"])
        learning_rates_D.append(optimizer_D.param_groups[0]["lr"])

        # # 更新学习率
        # scheduler_G.step()
        # scheduler_D.step()

        # 计算每个epoch上的评估指标
        real_batch = imgs[0: opt.batch_size]
        generated_batch = gen_imgs[0: opt.batch_size]
        real_batch = real_batch.detach().cpu().numpy().transpose(0, 2, 3,
                                                                 1)  # 能否直接用张量计算ssim，通过调用skimage→不能，这个库必须先转为numpy
        generated_batch = generated_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)#返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
        ssim_score = ssim(real_batch, generated_batch, multichannel=True, channel_axis=3, data_range=255)
        ssim_scores.append(ssim_score)
        psnr_score = psnr(real_batch, generated_batch,
                          data_range=255)  # The new argument is called channel_axis. This is the array axis along which your channels vary. Since you have a (256, 256, 3) image, the value should be 2 (or -1 which means “last”).
        psnr_scores.append(psnr_score)

        # 自己用于实验验证的部分，单张图片输入进行指标计算和整个批次放进去进行训练的效果是一样的
        for item in range(opt.batch_size):
            ssim_score_trail = ssim(real_batch[item], generated_batch[item], multichannel=True, channel_axis=2,
                                    data_range=1)
            ssim_scores_trail.append(ssim_score)
            psnr_score_trail = psnr(real_batch[item], generated_batch[item], data_range=1)
            psnr_scores_trail.append(psnr_score_trail)

        #     #计算和处理日志生成矩阵
        #     metrics = torch_fidelity.calculate_metrics(
        #         input1=torch_fidelity.GenerativeModelModuleWrapper(generator, opt.latent_dim, opt.noise_type, 0),
        #         input1_model_num_samples=opt.num_samples_for_metrics,
        #         input2=custom_dataset,
        #         isc=True,
        #         fid=True,
        #         kid=True,
        #         ppl=True,
        #         ppl_epsilon=1e-2,
        #         ppl_sample_similarity_resize=64,
        #     )
        #
        #     # 通过日志记录矩阵
        step += 1
        #     for k, v in metrics.items():
        #         writer.add_scalar(f'metrics/{k}', v, global_step=step)
        #
        #
        #     # save the generator if it improved
        #     if metric_greater_cmp(metrics[leading_metric], last_best_metric):
        #         print(f'Leading metric {opt.leading_metric} improved from {last_best_metric} to {metrics[leading_metric]}')
        #
        #         last_best_metric = metrics[leading_metric]
        #
        #         dummy_input = torch.zeros(1, opt.latent_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
        #         torch.jit.save(torch.jit.trace(generator, (dummy_input,)), os.path.join(opt.dir_logs, 'generator.pth'))
        #
        #
        # print(f'Training finished; the model with best {opt.leading_metric} value ({last_best_metric})')
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
            epoch, opt.n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item()))

        # 保存雪碧图
        if epoch % opt.sample_interval == 0:  # 即50个批次输出一次雪碧图
            save_image(gen_imgs.data[:opt.batch_size], "data/generated_images/%d.png" % epoch, nrow=5, normalize=True)

        writer.add_scalar('G lr', optimizer_G.param_groups[0]["lr"], step)
        writer.add_scalar('D lr', optimizer_D.param_groups[0]["lr"], step)

    writer.add_scalar('D loss', d_loss.item(), epoch)
    writer.add_scalar('G loss', g_loss.item(), epoch)

    folder = r'data/generated_images/eopch_' + str(epoch)

    ssim_score = np.mean(ssim_scores)
    psnr_score = np.mean(psnr_scores)
    writer.add_scalar('ssim_score', ssim_score,
                      epoch)  # 对于SSIM，其取值范围在-1到1之间，其中1表示两幅图像完全相同，而-1表示两幅图像完全相反。一般来说，SSIM值越接近1，说明两幅图像的结构相似性越好，质量也越高。因此，较好的SSIM值应该接近1。
    writer.add_scalar('psnr_score', psnr_score,
                      epoch)  # 对于PSNR，其单位为分贝（dB），取值范围在0到50之间。一般来说，PSNR值越高，说明压缩后的图像相对于原始图像的失真越小，质量也越高。因此，较好的PSNR值应该高于30dB，而理想情况下应该接近50dB。
    print("第{}批次的ssim计算平均值为{}".format(epoch, ssim_score))
    print("第{}批次的psnr计算平均值为{}".format(epoch, psnr_score))

    ssim_score_trail = np.mean(ssim_scores_trail)
    psnr_score_trail = np.mean(psnr_scores_trail)
    writer.add_scalar('ssim_score_trail', ssim_score_trail,
                      epoch)  # 对于SSIM，其取值范围在-1到1之间，其中1表示两幅图像完全相同，而-1表示两幅图像完全相反。一般来说，SSIM值越接近1，说明两幅图像的结构相似性越好，质量也越高。因此，较好的SSIM值应该接近1。
    writer.add_scalar('psnr_score_trail', psnr_score_trail,
                      epoch)  # 对于PSNR，其单位为分贝（dB），取值范围在0到50之间。一般来说，PSNR值越高，说明压缩后的图像相对于原始图像的失真越小，质量也越高。因此，较好的PSNR值应该高于30dB，而理想情况下应该接近50dB。
    print("第{}批次的ssim_trail计算平均值为{}".format(epoch, ssim_score_trail))
    print("第{}批次的psnr_trail计算平均值为{}".format(epoch, psnr_score_trail))

    # 保存单张图片，每一百批次保存一次
    if (epoch + 1) % 100 == 0:
        if not os.path.exists(folder):
            os.makedirs(folder)
        for j in range(opt.batch_size):
            save_image(gen_imgs.data[j], folder + "/" + str(j) + ".png")  # 每100批次中，最后一次生成的batch_size张图片

        # 保存断点，每100批次保存一次
        checkpoint = {"model_state_dict_G": generator.state_dict(),
                      "optimizer_state_dict_G": optimizer_G.state_dict(),
                      # 'scheduler_state_dict_G': scheduler_G.state_dict(),
                      "model_state_dict_D": discriminator.state_dict(),
                      "optimizer_state_dict_D": optimizer_D.state_dict(),
                      # 'scheduler_state_dict_D': scheduler_D.state_dict(),
                      "epoch": epoch,
                      "step": step,
                      }
        path_checkpoint = "{}/checkpoint_model.pth".format(opt.save_path)
        torch.save(checkpoint, path_checkpoint)

if opt.eraly_train is False:
    torch.save(generator.state_dict(), opt.save_path + '/' + "generator" + '.pth')
    torch.save(discriminator.state_dict(), opt.save_path + '/' + "discriminator" + '.pth')

# 关闭SummaryWriter
writer.close()
