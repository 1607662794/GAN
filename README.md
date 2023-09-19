# GAN

 **[demo.py](demo.py) ：使用CSDN的代码，然后在其上进行更改，实现数据图片的扩增→修改至自己的数据集，其生成的图片存放于 [generated_images](data\generated_images) 文件夹中。参考网页：http://t.csdn.cn/XY9El**

 [3_1. 基础GAN的代码实现  日月光华.ipynb](3_1. 基础GAN的代码实现  日月光华.ipynb) ：B站上的up主的代码

 [基础GAN的代码实现demo.ipynb](基础GAN的代码实现demo.ipynb) ：将B站up的代码改成自己的数据集后的代码，但是效果也不咋地



 [DCGAN生成混凝土图片.ipynb](DCGAN生成混凝土图片.ipynb) ：使用chat帮我写的DCGAN变体的代码，但是跑出来的总是噪点

 [换一个混凝土结构的文件夹试试.ipynb](换一个混凝土结构的文件夹试试.ipynb) ：AI生成的GAN代码



![real_samples](E:\Code\GAN\real_samples.png)：之前使用GAN生成的效果不错的图

------

 [data](data) ：存放图片和数据的文件夹，里面包含了我的源文件数据集和生成的数据集

​	 [best_images](data\best_images) ：生成的图片但是效果很不好

​	 [generated_images-DCGAN生成的第一次照片，但是生成器与对抗器无法持平](data\generated_images-DCGAN生成的第一次照片，但是生成器与对抗器无法持平) ：修改自己的数据库后，还没调节学习率前，生成的图片效果感觉还行，但是生成器与对抗器无法 很好地对抗

​	 [MNIST](data\MNIST) ：经典数据库

​	 [Plaster_side](data\Plaster_side) ：自己的混凝土数据库

------

 [logs](logs) ：断点保存时的权值文件

------

 [pytorch_gan_metrics](pytorch_gan_metrics) ：当时想要用这个自定义库来计算FID等指标，但是并不好用

------

 [runs](runs) ：tensorboard日志文件

------



------

 [pt_inception-2015-12-05-6726825d.pth](pt_inception-2015-12-05-6726825d.pth) ： [DCGAN生成混凝土图片.ipynb](DCGAN生成混凝土图片.ipynb) ：所必须的文件，我复制了一份放在了C盘

------

 [generated_images](generated_images) ： [DCGAN生成混凝土图片.ipynb](DCGAN生成混凝土图片.ipynb) 生成的图片，但是效果很不好

