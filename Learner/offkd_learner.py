from Learner.baselearner import BaseLearner
import torch
import time
import os
from KnowledgeDistillation.baseKD import OFFKD, ONKD

class OFFKDLearner(BaseLearner):
    def __init__(self, model,pretrained_model, time_data,file_path, configs):
        super(OFFKDLearner,self).__init__(model,time_data,file_path,configs)
        self.optimizer = self.model.optim
        self.criterion = self.model.loss
        self.scheduler = self.model.scheduler
        self.pretrained_model=pretrained_model
        self.kd_criterion=OFFKD[configs['kd_type']](configs['temperature'])

    def run(self):
        print("Training {} epochs".format(self.configs['epochs']))

        eval_accuracy, eval_loss = 0.0, 0.0
        train_accuracy, train_loss = 0.0, 0.0
        best_eval_accuracy=0.0
        # Train
        for epoch in range(self.configs['start_epoch'], self.configs['epochs'] + 1):
            train_accuracy, train_loss = self._train(epoch)
            eval_accuracy, eval_loss = self._eval()
            self.scheduler.step()
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
        self.model.train()  # train모드로 설정
        self.pretrained_model.eval()

        running_loss = 0.0
        correct = 0
        num_training_data = len(self.train_loader.dataset)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(
                self.device)  # gpu로 올림
            self.optimizer.zero_grad()  # optimizer zero로 초기화

            output = self.model(data)
            with torch.no_grad():
                cumbersome_output= self.pretrained_model(data)

            hard_loss = self.criterion(output, target)  # 결과와 target을 비교하여 계산

            soft_loss=self.kd_criterion(output,cumbersome_output)#KD
            loss=hard_loss+soft_loss


            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss.backward(retain_graph=True)  # 역전파

            # prune 이후 optimizer step
            self.optimizer.step()

            running_loss += loss.item()
            if batch_idx % self.log_interval == 0:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(
                    data), num_training_data, 100.0 * batch_idx / len(self.train_loader), loss.item()/(batch_idx+1)), end='')

        running_loss /= num_training_data
        tok = time.time()
        running_accuracy = 100.0 * correct / float(num_training_data)
        print('\nTrain Loss: {:.6f}'.format(running_loss), 'Learning Time: {:.1f}s'.format(
            tok-tik), 'Accuracy: {}/{} ({:.2f}%)'.format(correct, num_training_data, 100.0*correct/num_training_data))
        return running_accuracy, running_loss

    def _eval(self):
        self.model.eval()
        eval_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx,(data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                eval_loss += loss.item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        eval_loss = eval_loss / (batch_idx+1)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            eval_loss, correct, len(self.test_loader.dataset),
            100.0 * correct / float(len(self.test_loader.dataset))))
        eval_accuracy = 100.0*correct/float(len(self.test_loader.dataset))

        return eval_accuracy, eval_loss
