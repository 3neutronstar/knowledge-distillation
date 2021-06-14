from Learner.baselearner import BaseLearner
import torch
import time
import os
from KnowledgeDistillation.baseKD import ENSEMBLEKD

class EnsembleLearner(BaseLearner):
    def __init__(self, models, time_data,file_path, configs):
        super(EnsembleLearner,self).__init__(models,time_data,file_path,configs)
        # model = list of models
        # criterion
        self.optimizer=list()
        self.scheduler=list()
        self.criterion=list()
        for m in self.model:
            self.optimizer.append( m.optim)
            self.scheduler.append( m.scheduler)
            self.criterion.append(m.loss)
        self.kd_criterion=ENSEMBLEKD[configs['kd_type']]()
        self.classification_loss=list()
        self.model_output=list()

    def run(self):
        print("Training {} epochs".format(self.configs['epochs']))

        eval_accuracy, eval_loss = 0.0, 0.0
        train_accuracy, train_loss = 0.0, 0.0
        best_eval_accuracy=0.0
        # Train
        for epoch in range(self.configs['start_epoch'], self.configs['epochs'] + 1):
            train_accuracy, train_loss = self._train(epoch)
            eval_accuracy, eval_loss = self._eval()
            for scheduler in self.scheduler:
                scheduler.step()
            loss_dict = {'train': train_loss, 'eval': eval_loss}
            accuracy_dict = {'train': train_accuracy, 'eval': eval_accuracy}
            self.logWriter.add_scalars('loss', loss_dict, epoch)
            self.logWriter.add_scalars('accuracy', accuracy_dict, epoch)

            self.early_stopping(eval_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            if best_eval_accuracy<eval_accuracy:
                best_eval_accuracy=eval_accuracy
        print("Best Accuracy in evaluation: {:.2f}".format(best_eval_accuracy) )

    def _train(self, epoch):
        tik = time.time()
        for m in self.model:
            m.train()  # train모드로 설정
        running_loss = 0.0
        correct = 0
        num_training_data = len(self.train_loader.dataset)
        torch.autograd.set_detect_anomaly(True)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.model_output=list()
            self.classification_loss=list()
            data, target = data.to(self.device), target.to(
                self.device)  # gpu로 올림

            for m,criterion in zip(self.model,self.criterion):
                output=m(data)
                classification_loss = criterion(output, target)
                self.classification_loss.append(classification_loss)
                self.model_output.append(output)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            
            for idx,(optim,c_l) in enumerate(zip(self.optimizer,self.classification_loss)):
                optim.zero_grad()
                model_loss=c_l+self.kd_criterion(idx,self.model_output)
                c_l.backward(retain_graph=True)  # 역전파
                optim.step()
                running_loss += model_loss.item()

            if batch_idx % self.log_interval == 0:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(
                    data), num_training_data, 100.0 * batch_idx / len(self.train_loader), model_loss.item()), end='')
        correct/=float(len(self.model))
        running_loss /= (batch_idx+1)
        tok = time.time()
        running_accuracy = 100.0 * correct / float(num_training_data)
        print('\nTrain Loss: {:.6f}'.format(running_loss), 'Learning Time: {:.1f}s'.format(
            tok-tik), 'Accuracy: {}/{} ({:.2f}%)'.format(correct, num_training_data, 100.0*correct/num_training_data))
        return running_accuracy, running_loss

    def _eval(self):
        for m in self.model:
            m.eval()
        eval_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                for m,criterion in zip(self.model,self.criterion):
                    output = m(data)
                    loss = criterion(output, target)
                    eval_loss += loss.item()
                    # get the index of the max log-probability
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
        correct/=float(len(self.model))
        eval_loss = eval_loss / len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            eval_loss, correct, len(self.test_loader.dataset),
            100.0 * correct / float(len(self.test_loader.dataset))))
        eval_accuracy = 100.0*correct/float(len(self.test_loader.dataset))

        return eval_accuracy, eval_loss
