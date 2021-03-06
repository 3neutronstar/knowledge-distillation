import os
import torch
from Learner.baselearner import BaseLearner, ClassicLearner
from KnowledgeDistillation.baseKD import *
import time
import logging
import torch.nn as nn
CUSTOM_LOSS={
    'cs-kd':OFFKD['softtarget'],
    'baseline':None,
    'pearson':OURS['ver1'],
    'kldiv':OURS['ver2'],
}

def set_logging_defaults(logdir):
    # set basic configuration for logging
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])

    # log cmdline argumetns
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))


class SelfKDLearner(ClassicLearner):
    def __init__(self, model, time_data,file_path, configs):
        super(SelfKDLearner,self).__init__(model,time_data,file_path,configs)
        this_path=os.path.join(file_path,'training_data',time_data)
        set_logging_defaults(this_path)
        self.logger = logging.getLogger('main')
        if CUSTOM_LOSS[configs['custom_loss']] is None:
            self.KDCustomLoss=False
        else:
            self.KDCustomLoss=True
            self.kdloss=CUSTOM_LOSS[configs['custom_loss']](self.configs['temperature'])
        self.criterion = self.model.loss

    def run(self):
        self.logger.info(self.configs)
        super().run()
        self.logger = logging.getLogger('best')
        self.logger.info('[Acc {:.3f}]'.format(self.best_eval_accuracy))


    def _train(self,epoch):
        tik = time.time()
        self.model.train()  # train모드로 설정
        train_cls_loss = 0.0
        train_loss=0.0
        total=0
        
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            batch_size = inputs.size(0)

            if not self.KDCustomLoss: # baseline
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                train_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).sum().float().cpu()
            else:
                if self.configs['custom_loss']=='cs-kd':
                    targets_ = targets[:batch_size//2].detach().clone()
                    outputs = self.model(inputs[:batch_size//2])
                    loss = self.criterion(outputs, targets_)
                    train_loss += loss.item()

                    with torch.no_grad():
                        outputs_cls = self.model(inputs[batch_size//2:])
                    cls_loss = self.kdloss(outputs, outputs_cls.detach())
                    loss += self.configs['lambda'] * cls_loss
                    train_cls_loss += cls_loss.item()
                else: # covariance loss (pearson,kl_div)
                    targets_=targets
                    outputs=self.model(inputs)
                    loss = self.criterion(outputs, targets_)
                    train_loss += loss.item()
                    if epoch>self.configs['epochs']*self.configs['start_epoch_rate']:
                        cls_loss=self.kdloss(outputs,targets_)
                        for param in self.model.parameters():
                            param.requires_grad = False
                        for p in self.model.classifier.parameters():
                            p.requires_grad =True
                        cls_loss.backward(retain_graph=True)
                        for param in self.model.parameters():
                            param.requires_grad = True
                                                
                        loss += self.configs['lambda'] * cls_loss.detach()
                        train_cls_loss += cls_loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets_.size(0)
                correct += predicted.eq(targets_.data).sum().float().cpu()

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, total
                , len(self.train_loader.dataset), 100.0 * total / len(self.train_loader.dataset), train_loss/(batch_idx+1)), end='')
        tok=time.time()
        running_accuracy=100.0*float(correct)/float(total)
        train_loss/=(batch_idx+1)
        #logger
        self.logger = logging.getLogger('train')
        self.logger.info('[Epoch {}] [Loss {:.3f}] [KDCustomLoss {:.3f}] [Acc {:.3f}] [Learning Time:{:.2f}]'.format(epoch,
        train_loss,
        train_cls_loss/(batch_idx+1),
        100.*correct/total,tok-tik))
        self.logWriter.add_scalar('loss/cls_loss',train_cls_loss,epoch)
        return running_accuracy, train_loss