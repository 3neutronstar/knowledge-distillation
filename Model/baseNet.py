
def get_hyperparams(nn_type,KD_MODE=False):
    if nn_type == 'lenet5':
        dataset = 'mnist'
        epochs = 60
        lr=1e-2
        momentum=0.9
    elif nn_type == 'vgg16':
        dataset = 'cifar10'
        epochs = 300
        lr=1e-2
        momentum=0.9
    elif nn_type=='lenet300_100':
        dataset = 'mnist'
        epochs = 60
        lr=1e-2
        momentum=0.9
    elif 'resnet' in nn_type:
        dataset='cifar10'
        lr=1e-1
        epochs=200
        momentum=0.9
    elif nn_type=='convnet':
        dataset = 'cifar10'
        epochs = 200
        lr=1e-2
        momentum=0.9
    elif nn_type=='alexnet':
        dataset='cifar10'
        epochs=200
        lr=1e-2
        momentum=0.9
    else:
        print("No algorithm available")
        raise NotImplementedError
    
    if KD_MODE==True:
        dataset='imagenet'
    

    return dataset,epochs,lr,momentum



class BaseNet():
    def __init__(self,configs):
        
        if configs['dataset']=='cifar10' or configs['dataset']=='mnist':
            configs['num_classes']=10
        elif configs['dataset']=='cifar100':
            configs['num_classes']=100
        else:#imagenet
            configs['num_classes']=1000
        
        if configs['model'] == 'lenet5':
            from Model.lenet5 import LeNet5
            model = LeNet5(configs).to(configs['device'])
        elif configs['model'][:3] == 'vgg':
            from Model.vgg import VGG
            model = VGG(configs).to(configs['device'])
            # print(model)
        elif configs['model']=='lenet300_100':
            from Model.lenet300_100 import LeNet_300_100
            model = LeNet_300_100(configs).to(configs['device'])
        elif configs['model'][:6]=='resnet':
            from Model.resnet import ResNet
            model = ResNet(configs).to(configs['device'])
        elif configs['model']=='convnet':
            from Model.convnet import ConvNet
            model = ConvNet(configs).to(configs['device'])
        elif configs['model']=='alexnet':
            from Model.alexnet import AlexNet
            model = AlexNet(configs).to(configs['device'])
        else:
            print("No Model")
            raise NotImplementedError
        self.model=model

        if 'kd' in configs['mode']:
            if configs['pretrained_model']=='vgg16':
                from torchvision.models import vgg16
                pretrained_model=vgg16(pretrained=False)
            if configs['pretrained_model']=='vgg16bn':
                from torchvision.models import vgg16_bn
                pretrained_model=vgg16_bn(pretrained=False)
            if configs['pretrained_model']=='resnet18':
                from torchvision.models import resnet18
                pretrained_model=resnet18(pretrained=False)

            import torch
            pretrained_model.load_state_dict(torch.load('./pretrained_data/{}_{}.pth'.format(configs['model'],configs['dataset'])))
            self.pretrained_model=pretrained_model