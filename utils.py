import json
import os
import random
import torch
import numpy as np
import torchvision
from torch.utils.data import Sampler, Dataset
from collections import defaultdict


def load_params(configs, file_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    ''' replay_name from flags.replay_name '''
    with open(os.path.join(current_path, 'training_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
    return configs


def save_params(configs, time_data):
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, 'training_data', '{}.json'.format(time_data)), 'w') as fp:
        json.dump(configs, fp, indent=2)

def load_model(model,file_path,file_name):
    model.load_state_dict(torch.load(os.path.join(file_path,'training_data','checkpoint_'+file_name+'.pt')))
    return model


class EarlyStopping:
    """주어진 patience 이후로 train loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, file_path,time_data,config,patience=7, verbose=False, delta=0,):
        """
        Args:
            patience (int): train loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 train loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.config=config
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        if self.config['mode']=='train':
            self.path= os.path.join(file_path,'training_data','{}_{}.pt'.format(self.config['model'],self.config['dataset']))
        elif 'kd' in self.config['mode']:
            self.path = os.path.join(file_path,'training_data',time_data,'distilled_data','{}_{}.pt'.format(self.config['model'],self.config['dataset']))

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:

            self.counter += 1
            if self.config['earlystop']==True:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            print(
                f'Eval loss not decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(
                f'Eval loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if type(model)==list:
            print("Saving Ensemble")
            torch.save(model[0].state_dict(), self.path+'0')
            torch.save(model[0].state_dict(), self.path+'1')
            
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



        
class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, torch.utils.datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]
