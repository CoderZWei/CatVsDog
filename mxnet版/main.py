from mxnet.gluon import data as gdata,model_zoo,nn
from mxnet.gluon import loss as gloss
import mxnet as mx
from mxnet import gluon,autograd
import mxnet.ndarray as nd
import os
import pandas as pd
from mxnet.gluon.data import dataset
from PIL import Image

#定义参数
batch_size=32
num_epochs=150
lr=0.01
wd=1e-4
ctx=mx.cpu()
train_dir='catvsdog/train_data/'
test_dir='catvsdog/test/data/'
model_dir='model/2_gpu/'

with mx.Context(ctx):
    #数据加载
    data_loader=gdata.vision.ImageFolderDataset(train_dir,flag=1)
    normalize=gdata.vision.transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )
    train_augs = gdata.vision.transforms.Compose([
        gdata.vision.transforms.RandomResizedCrop(224,scale=(0.75,1.0),ratio=(1.0,1.0)),
        gdata.vision.transforms.RandomFlipLeftRight(),
        gdata.vision.transforms.ToTensor(),
        normalize])

    test_augs = gdata.vision.transforms.Compose([
        gdata.vision.transforms.Resize(224),
        gdata.vision.transforms.ToTensor(),
        normalize])
    loader=gluon.data.DataLoader
    train_data=loader(data_loader.transform_first(train_augs),batch_size=batch_size,shuffle=True,last_batch='keep')

    #定义和初始化模型
    resnet=mx.gluon.model_zoo.vision.resnet18_v2()
    resnet.load_parameters('resnet18_v2-8aacf80f.params',ctx=ctx)

    def Classifier():
        net = nn.HybridSequential()
        net.add(nn.Dense(1024, activation="relu"))
        net.add(nn.Dropout(.5))
        net.add(nn.Dense(512, activation="relu"))
        net.add(nn.Dropout(.5))
        net.add(nn.Dense(2,activation='sigmoid'))
        return net

    net=nn.HybridSequential()
    appendnet=Classifier()
    net.feature=nn.HybridSequential()
    net.feature.add(resnet.features)
    net.output=nn.HybridSequential()
    net.output.add(appendnet)
    net.output.initialize(init=mx.init.Xavier(),ctx=mx.cpu())

    #定义损失函数
    loss=gloss.SoftmaxCrossEntropyLoss()
    '''
    #训练模型
    trainer=gluon.Trainer(net.output.collect_params(),'sgd',{'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    for epoch in range(num_epochs):
        for i,(img,label) in enumerate(train_data):
            with autograd.record():
                img=img.as_in_context(ctx)
                label = label.as_in_context(ctx)
                print(label)
                result=net(img)
                predict=nd.argmax(result,axis=1)
                print('predict:',predict)
                l=loss(result,label)
            l.backward()
            trainer.step(batch_size=batch_size)
            l_sum=l.sum()
            print('epoch:',epoch)
            print('loss:',l_sum)
            net.save_parameters(model_dir+ "epoch_" + str(epoch) + ".params")
    '''
    #测试模型
    test_img_list=os.listdir(test_dir)
    test_img_list.sort(key=lambda x:int(x[:-4]))
    net.load_parameters(model_dir+'epoch_149.params')
    labels=nd.ones(2000)
    for i,img_index in enumerate(test_img_list):
        img = Image.open(test_dir + img_index)
        img=nd.array(img)
        img = test_augs(img)
        img=nd.expand_dims(img,axis=0)
        labels[i]=nd.argmax(net(img),axis=1)
    lab = pd.Series(labels.asnumpy(), index=test_img_list)
    print(lab)
    lab.to_csv('submission0.csv')
