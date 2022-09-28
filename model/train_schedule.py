import time
import model_defination
import load_data
import configuration
import json
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'model'))


class ModelTrain:
    def __init__(self,
                 para_dict,
                 net,
                 transform,
                 forcing_load_from_pic=True) -> None:
        self.save_location = para_dict['save_location']
        self.lr = para_dict['learning_rate']
        self.num_epochs = para_dict['num_epochs']
        self.batch_size = para_dict['batch_size']
        self.drop_prob = para_dict['drop_prob']
        self.selected_remarks = para_dict['selected_remarks']
        self.train_percentage = para_dict['train_percentage']
        self.step_size = para_dict['step_size']
        self.gamma = para_dict['gamma']
        self.aloud = para_dict['aloud']
        self.net = net
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, self.step_size, self.gamma)
        self.current_time = time.strftime("%Y_%m_%d_%H_%M_%S",
                                          time.localtime())
        self.log_path = os.path.join(para_dict['log_path'], self.current_time)
        self.writer = SummaryWriter(self.log_path)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.cmedata = load_data.CMEdata(self.save_location,
                                         self.selected_remarks,
                                         self.train_percentage, transform)
        self.forcing_load_from_pic = forcing_load_from_pic

    def __create_folder(self, path_to_folder):
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)

    def __get_dataloader(self):
        self.cmedata.load_data(self.forcing_load_from_pic)
        train_dataset = self.cmedata.to_tensordataset(is_train=True)
        test_dataset = self.cmedata.to_tensordataset(is_train=False)
        train_iter = torch.utils.data.DataLoader(train_dataset,
                                                 self.batch_size,
                                                 shuffle=True)
        test_iter = torch.utils.data.DataLoader(test_dataset,
                                                self.batch_size,
                                                shuffle=True)
        return train_iter, test_iter

    def __save_epoch_info(self, train_info_path):
        df = pd.DataFrame(self.train_details_list)
        filename = os.path.join(train_info_path, 'epoch_info.xlsx')
        df.to_excel(filename)

    class _ModuleEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.nn.Module):
                return repr(obj.__class__)
            if isinstance(obj, torch.optim.lr_scheduler._LRScheduler):
                return repr(obj.__class__)
            return json.JSONEncoder.default(self, obj)

    def __save_para_info(self, train_info_path):
        pos = self.cmedata.train_label.sum().item()
        neg = self.cmedata.train_label.shape[0] - pos
        para = {}
        para['lr'] = self.lr
        para['num_epochs'] = self.num_epochs
        para['batch_size'] = self.batch_size
        para['drop_prob'] = self.drop_prob
        para['selected_remarks'] = self.selected_remarks
        para['train_percentage'] = self.train_percentage
        para['scheduler_step_size'] = self.step_size
        para['CME_count'] = pos
        para['No_CME_count'] = neg
        para['CME:NO CME'] = '{}:1'.format(pos / neg)
        para['Net'] = self.net
        para['lr_scheduler'] = self.scheduler
        filename = os.path.join(train_info_path, 'para.json')
        with open(filename, 'w') as f:
            json.dump(para, f, cls=ModelTrain._ModuleEncoder)

    def __save_model_info(self, model_info_path):
        filename = os.path.join(self.log_path, 'parameters.pkl')
        torch.save(self.net.state_dict(), filename)

    def save_info(self):
        self.__create_folder(self.log_path)
        self.__save_epoch_info(self.log_path)
        self.__save_para_info(self.log_path)
        self.__save_model_info(self.log_path)
        print('Save training detailed infomation to {}'.format(self.log_path))

    def evaluate_accuracy_on(self, X, y):
        """
        求在给定数据集上的正确率
        Arguments:
        ---------
        X : 数据集
        y : 数据集标签
        Returns:
        -------
        accuracy :在给定数据集上的正确率
        """

        num_accu, total = 0, 0
        X, y = X.to(self.device), y.to(self.device)
        self.net.eval()  #修改为评估模式
        num_accu += (torch.argmax(self.net(X), dim=1) == y).sum().item()
        self.net.train()  #修改为训练模式
        total += X.shape[0]
        accuracy = num_accu / total
        return accuracy

    def __evaluate_accuracy_on_testiter(self, test_iter):
        num_accu, total = 0, 0
        self.net.eval()  # 修改为评估模式
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(self.device), y.to(self.device)
                num_accu += (torch.argmax(self.net(X),
                                          dim=1) == y).sum().item()
                total += X.shape[0]
        self.net.train()
        return num_accu / total

    def __evaluate_on_testiter(self, test_iter):
        num_accu, total, test_l_sum, batch_count = 0, 0, 0, 0
        self.net.eval()  # 修改为评估模式
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                l = self.loss(y_hat, y)
                test_l_sum += l.cpu().item()
                num_accu += (torch.argmax(self.net(X),
                                          dim=1) == y).sum().item()
                total += X.shape[0]
                batch_count += 1
        self.net.train()
        return num_accu / total, test_l_sum / batch_count

    def __log(self, train_loss, train_accu, test_loss, test_accu, lr,
              global_step):
        self.writer.add_scalar('loss/train', train_loss, global_step)
        self.writer.add_scalar('loss/test', test_loss, global_step)
        self.writer.add_scalar('accu/train', train_accu, global_step)
        self.writer.add_scalar('accu/test', test_accu, global_step)
        self.writer.add_scalar('lr', lr, global_step)

    def early_stop(self, patience, metric, mode):
        decide_stop = False
        scores = torch.tensor(
            [epoch_detail[metric] for epoch_detail in self.train_details_list])
        best_score_idx = scores.argmax() if mode == 'max' else scores.argmin()
        if best_score_idx == len(scores) - 1:
            self.__save_model_info(self.log_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(
                metric, scores[best_score_idx]),
                  file=sys.stderr)
        if len(scores) - best_score_idx > patience:
            print(
                "<<<<<< {} without improvement in {} epoches, early stopping >>>>>>"
                .format(metric, patience),
                file=sys.stderr)
            decide_stop = True
        return decide_stop

    def fit(self, patience=5, metric='epoch_test_accu', mode='max'):
        train_iter, test_iter = self.__get_dataloader()
        self.net = self.net.to(self.device)
        self.net.train()
        print('training on:', self.device)
        batch_count = 0
        # 计算总的iteration数量
        total_iterations = self.num_epochs * \
            (int(self.cmedata.train_size/self.batch_size)+1)
        print('----------------------')
        print('Begin training:')
        print('total {} epoches, {} iterations each epoch'.format(
            self.num_epochs,
            int(self.cmedata.train_size / self.batch_size) + 1))
        # 如果在终端输出更多信息，则tqdm的total应为总的iteration数，在每个batch结束后更新pbar
        # 如果不需要输出更多信息，则tqdm的total应为总的epoch数，在每个epoch结束后更新pbar
        if self.aloud:
            pbar = tqdm(total=total_iterations)
        else:
            pbar = tqdm(total=self.num_epochs)
        pbar.set_description('epoch {} iteration {}'.format(1, 1))
        train_start = time.time()
        # 记录每次epoch情况的列表
        self.train_details_list = []
        try:
            for epoch in range(self.num_epochs):
                # 分别表示每epoch训练总损失，每epoch训练正确数，每epoch样本总数
                train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
                current_epoch_start = time.time()
                iteration_count = 0
                for X, y in train_iter:
                    iteration_count += 1
                    X = X.to(self.device)
                    y = y.to(self.device)
                    y_hat = self.net(X)
                    l = self.loss(y_hat, y)
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    train_l_sum += l.cpu().item()
                    train_acc_sum += (y_hat.argmax(
                        dim=1) == y).sum().cpu().item()
                    n += y.shape[0]
                    batch_count += 1
                    if self.aloud:
                        pbar.set_description('epoch {} iteration {}'.format(
                            epoch + 1, iteration_count))
                        pbar.update(1)
                test_accu, test_loss = self.__evaluate_on_testiter(test_iter)
                epoch_train_loss = train_l_sum / batch_count
                epoch_train_accu = train_acc_sum / n
                # 如果aloud为False，则每个epoch之后才更新一次pbar
                if not self.aloud:
                    pbar.set_description('epoch {} iteration {}'.format(
                        epoch + 1, iteration_count))
                    pbar.update(1)
                epoch_time = time.time() - current_epoch_start
                total_time = time.time() - train_start
                epoch_detail_dict = {
                    'epoch_num': epoch,
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_train_accu': epoch_train_accu,
                    'epoch_test_loss': test_loss,
                    'epoch_test_accu': test_accu,
                    'epoch_time': epoch_time,
                    'total_time': total_time
                }
                self.__log(epoch_train_loss, epoch_train_accu, test_loss,
                           test_accu, self.optimizer.param_groups[0]['lr'],
                           epoch)
                self.train_details_list.append(epoch_detail_dict)
                pbar.set_postfix(train_loss=epoch_train_loss,
                                 train_accu=epoch_train_accu,
                                 test_loss=test_loss,
                                 test_accu=test_accu)
                self.scheduler.step()
                if self.early_stop(patience, metric, mode):
                    break
        except KeyboardInterrupt:
            print('Training manually interrupted')
        finally:
            pbar.close()
            print('Finish training')
            self.save_info()
            self.writer.close()

    def infer(self, path: str):
        """
        给定图片路径，判断是否是CME
        Arguments:
        ---------
        pic:图片路径
        
        Returns:
        -------
        resu:是否为CME
        """
        img = Image.open(path).convert('L')
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        self.net.eval()
        y = self.net(img)
        #resu = torch.argmax(self.net(img), dim=1)
        return y


if __name__ == '__main__':
    trans = load_data.CenterCrop('NCHW')
    net = model_defination.vgg19(
        drop_prob=configuration.para_dict['drop_prob'])
    modeltrain = ModelTrain(configuration.para_dict,
                            net,
                            trans)
    modeltrain.fit()
    modeltrain.save_info()
    resu = modeltrain.infer(
        r'D:\Programming\CME_data\CME\Halo\20130830_032405_lasc2rdf_aia193rdf.png'
    )
    print(resu)
