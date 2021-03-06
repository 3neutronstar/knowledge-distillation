from torchvision import datasets
import torchvision.transforms as transforms
import torch
import sys
from six.moves import urllib 
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler, Subset
from utils import DatasetWrapper, PairBatchSampler
def load_dataset(configs):
    if sys.platform == 'linux':
        data_save_path='../data/dataset'
    elif sys.platform =='win32':
        data_save_path='..\data\dataset'
    else:
        data_save_path='../data/dataset'
    if configs['dataset'] == 'mnist':
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST(root=data_save_path, train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root=data_save_path, train=False,
                                        download=False, transform=transform)

    elif configs['dataset'] == 'cifar100':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        train_transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        test_transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_data = datasets.CIFAR100(root=data_save_path, train=True,
                                       download=True, transform=train_transform)
        test_data = datasets.CIFAR100(root=data_save_path, train=False,
                                      download=False, transform=test_transform)

    elif configs['dataset'] == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        train_transform=transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        train_data = datasets.CIFAR10(root=data_save_path, train=True,
                                      download=True, transform=train_transform)
        test_data = datasets.CIFAR10(root=data_save_path, train=False,
                                     download=False, transform=test_transform)

    elif configs['dataset']=='imagenet':
        data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
        train_data=datasets.ImageNet(root=data_save_path,train=True,transform=data_transforms['train'],download=True )
        test_data=datasets.ImageNet(root=data_save_path,train=False,transform=data_transforms['test'],download=False )

    elif configs['dataset']=='fashionmnist':
        train_transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_data=datasets.FashionMNIST(root=data_save_path, download=True, train=True, transform=train_transform)
        test_data=datasets.FashionMNIST(root=data_save_path, download=False, train=False, transform=test_transform)
    
    return train_data, test_data

def base_data_loader(train_data,test_data,configs):
    if configs['device'] == 'gpu':
        pin_memory = True
        # pin_memory=False
    else:
        pin_memory = False
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=configs['batch_size'],
                                                    shuffle=True,
                                                    pin_memory=pin_memory,
                                                    num_workers=configs['num_workers'],
                                                    )
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=configs['batch_size'],
                                                   shuffle=True,
                                                   pin_memory=pin_memory,
                                                   num_workers=configs['num_workers'],
                                                   )

    print("Using Datasets: ", configs['dataset'])
    return train_data_loader, test_data_loader

def pair_data_loader(train_data,test_data,configs):
    if configs['device'] == 'gpu':
        pin_memory = True
        # pin_memory=False
    else:
        pin_memory = False
        # Sampler
    if configs['sample'] == 'default':
        get_train_sampler = lambda d: BatchSampler(RandomSampler(d), configs['batch_size'], False)
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), configs['batch_size'], False)

    elif configs['sample'] == 'pair':
        get_train_sampler = lambda d: PairBatchSampler(d, configs['batch_size'])
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), configs['batch_size'], False)
    train_data=DatasetWrapper(train_data)
    test_data=DatasetWrapper(test_data)
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_sampler=get_train_sampler(train_data),
                                                    pin_memory=pin_memory,
                                                    num_workers=configs['num_workers'],
                                                    )
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                    batch_sampler=get_test_sampler(test_data),
                                                    pin_memory=pin_memory,
                                                    num_workers=configs['num_workers'],
                                                    )

    return train_data_loader, test_data_loader
    
def data_loader(configs):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    train_data, test_data = load_dataset(configs)


    if configs['mode']=='train_selfkd' and configs['custom_loss']=='cs-kd':
        train_data_loader,test_data_loader=pair_data_loader(train_data,test_data,configs)
    elif configs['mode']=='train' or configs['mode']=='test' or 'kd' in configs['mode']:
        train_data_loader, test_data_loader=base_data_loader(train_data, test_data,configs)
    
    return train_data_loader, test_data_loader
