import os

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.metrics import iou, accuracy, precision, recall
from utils.metrics import threshold
from utils.read_dataset import ImageDataset, read_txt
from utils.loss_function import BinaryCrossEntropy, BinaryDiceLoss

from models.UNet import UNet

# ----------------- Hyperparameter Settings -------------------------------------
batch_size = 16
total_epoch = 120
learning_rate = 0.001
model_list = ['U-Net', ]
loss_list = ['Cross Entropy Loss', ]
# --------------------------------------------------------------------------------


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_dict = {
    'U-Net': UNet(in_ch=3, out_ch=1),
}
loss_dict = {
    'Cross Entropy Loss': BinaryCrossEntropy(),
    'Dice Loss': BinaryDiceLoss(),
}

trainset = ImageDataset(
    image_path=read_txt(txt_path=r'datasets\images_train.txt'),
    label_path=read_txt(txt_path=r'datasets\labels_train.txt'),
    flag='train',
)
valset = ImageDataset(
    image_path=read_txt(txt_path=r'datasets\images_val.txt'),
    label_path=read_txt(txt_path=r'datasets\labels_val.txt'),
    flag='val',
)
testset = ImageDataset(
    image_path=read_txt(txt_path=r'datasets\images_test.txt'),
    label_path=read_txt(txt_path=r'datasets\labels_test.txt'),
    flag='test',
)
train_loader = DataLoader(trainset, batch_size=batch_size, drop_last=False, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True)
test_loader = DataLoader(testset, batch_size=1, drop_last=False, shuffle=False, pin_memory=True)


class Trainer(object):
    def __init__(self, model_name, loss_name):
        self.model_name = model_name
        self.model = models_dict[model_name].to(device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.loss_fn = loss_dict[loss_name].to(device)

        self.train_loss = 0.0
        self.train_iou = 0.0
        self.train_accuracy = 0.0

        self.best_iou = 0.0
        self.current_epoch = 1
        self.epoch_information = pd.DataFrame({})

        self.experimental_result_path = os.path.join(r'./results', model_name, loss_name)
        self.epoch_information_path = os.path.join(self.experimental_result_path, 'epoch_information.csv')
        self.test_information_path = os.path.join(self.experimental_result_path, 'test_information.csv')

        self.current_state_path = os.path.join(self.experimental_result_path, 'current_state.pth.tar')
        self.best_state_path = os.path.join(self.experimental_result_path, 'best_state.pth.tar')

        self.val_visual_result = os.path.join(self.experimental_result_path, 'visualization', 'valset')
        self.train_visual_result = os.path.join(self.experimental_result_path, 'visualization', 'trainset')
        self.val_visual_result = os.path.join(self.experimental_result_path, 'visualization', 'valset')
        self.test_visual_result = os.path.join(self.experimental_result_path, 'visualization', 'testset')

        if not os.path.exists(self.experimental_result_path):
            os.makedirs(self.experimental_result_path)
        elif os.path.exists(self.current_state_path):
            print("Loading current state: '{}'".format(self.current_state_path))
            current_state = torch.load(self.current_state_path)
            self.model.load_state_dict(current_state['model_parameters'])
            self.optimizer.load_state_dict(current_state['optimizer'])
            self.best_iou = current_state['best_iou']

            self.epoch_information = pd.read_csv(self.epoch_information_path)
            self.current_epoch = self.epoch_information['epoch'].size + 1

    def train(self):
        self.model.train()
        tbar = tqdm(train_loader, ncols=80)

        iou_sum = 0.0
        accuracy_sum = 0.0
        loss_sum = 0.0
        for i, (image_batch, label_batch) in enumerate(tbar):
            image = image_batch.to(device)
            target = label_batch.to(device)
            target = threshold(target, threshold=0.5)

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.loss_fn(output, target)
            loss_sum += loss.item()
            tbar.set_description('train loss: %.3f' % (loss_sum / (i + 1)))

            iou_sum += iou(output, target).item()
            accuracy_sum += accuracy(output, target).item()

            loss.backward()
            self.optimizer.step()
        tbar.close()

        self.train_loss = loss_sum / len(train_loader)
        self.train_iou = iou_sum / len(train_loader)
        self.train_accuracy = accuracy_sum / len(train_loader)
        print("trainset: iou:{:.3f}, accuracy: {:.3f}".format(self.train_iou, self.train_accuracy))

    def val(self):
        self.model.eval()
        tbar = tqdm(val_loader, ncols=80)

        iou_sum = 0.0
        accuracy_sum = 0.0
        loss_sum = 0.0
        for i, (image_batch, label_batch) in enumerate(tbar):
            image = image_batch.to(device)
            target = label_batch.to(device)
            target = threshold(target, threshold=0.5)

            with torch.no_grad():
                output = self.model(image)
            loss = self.loss_fn(output, target)
            loss_sum += loss.item()
            tbar.set_description('val loss: %.3f' % (loss_sum / (i + 1)))

            iou_sum += iou(output, target).item()
            accuracy_sum += accuracy(output, target).item()
        tbar.close()

        val_loss = loss_sum / len(val_loader)
        epoch_iou = iou_sum / len(val_loader)
        epoch_accuracy = accuracy_sum / len(val_loader)
        print("valset: iou:{:.3f}, accuracy: {:.3f}".format(epoch_iou, epoch_accuracy))
        if self.best_iou < epoch_iou:
            self.best_iou = epoch_iou
            best_state = {'model_parameters': self.model.state_dict()}
            torch.save(best_state, self.best_state_path)
            print('best state save successfully!')

        current_epoch_information = pd.DataFrame({
            'epoch': [self.current_epoch],
            'train_loss': [self.train_loss],
            'val_loss': [val_loss],
            'train_iou': [self.train_iou],
            'val_iou': [epoch_iou],
            'train_accuracy': [self.train_accuracy],
            'val_accuracy': [epoch_accuracy],
            'best_iou': [self.best_iou],
        })
        self.epoch_information = pd.concat([self.epoch_information, current_epoch_information],
                                           axis=0, ignore_index=True)
        self.epoch_information.to_csv(self.epoch_information_path, index=False)
        current_state = {
            'model_parameters': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
        }
        torch.save(current_state, self.current_state_path)
        print('current state save successfully!')

    def test(self):
        self.model.eval()
        tbar = tqdm(test_loader, ncols=80)

        iou_sum = 0.0
        accuracy_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        loss_sum = 0.0
        for i, (image_batch, label_batch) in enumerate(tbar):
            image = image_batch.to(device)
            target = label_batch.to(device)
            target = threshold(target, threshold=0.5)

            with torch.no_grad():
                output = self.model(image)
            loss = self.loss_fn(output, target)
            loss_sum += loss.item()
            tbar.set_description('test loss: %.3f' % (loss_sum / (i + 1)))

            iou_sum += iou(output, target).item()
            accuracy_sum += accuracy(output, target).item()
            precision_sum += precision(output, target).item()
            recall_sum += recall(output, target).item()
        tbar.close()

        test_loss = loss_sum / len(test_loader)
        epoch_iou = iou_sum / len(test_loader)
        epoch_accuracy = accuracy_sum / len(test_loader)
        epoch_precision = precision_sum / len(test_loader)
        epoch_recall = recall_sum / len(test_loader)
        print("testset: iou:{:.3f}, accuracy: {:.3f}, precision:{:.3f}, recall: {:.3f}".format(
            epoch_iou, epoch_accuracy, epoch_precision, epoch_recall))
        test_info = pd.DataFrame({
            'total_epoch': [total_epoch],
            'test_loss': [test_loss],
            'iou': [epoch_iou],
            'accuracy': [epoch_accuracy],
            'precision': [epoch_precision],
            'recall': [epoch_recall]
        })
        test_info.to_csv(self.test_information_path)
        self.plot_image(_dataset=testset, save_dir=self.test_visual_result)

    def run(self):
        print('start epoch: ', self.current_epoch)
        print('total epoch: ', total_epoch)
        for epoch in range(self.current_epoch, total_epoch + 1):
            self.current_epoch = epoch
            print("\nepoch %d: " % self.current_epoch)
            self.train()
            self.val()

        print("\nloading best checkpoint: '{}'".format(self.best_state_path))
        best_checkpoint = torch.load(self.best_state_path)
        self.model.load_state_dict(best_checkpoint['model_parameters'])

        self.test()

    def plot_image(self, _dataset, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        _data_loader = DataLoader(_dataset, batch_size=1, drop_last=False,
                                  shuffle=False, pin_memory=True)

        tbar = tqdm(_data_loader, ncols=80)
        for i, (image_batch, label_batch) in enumerate(tbar):
            image = image_batch.to(device)
            target = label_batch.to(device)
            target = threshold(target, threshold=0.5)

            with torch.no_grad():
                output = self.model(image)
            tbar.set_description('visual images: ')

            _iou = iou(output, target).item()
            predict = threshold(output, threshold=0.5)
            save_image(image, os.path.join(save_dir, str(i) + "_image" + ".jpg"))
            save_image(target, os.path.join(save_dir, str(i) + "_label" + ".jpg"))
            save_image(predict.float(), os.path.join(save_dir,
                                                     str(i) + "_predict_iou_" + str(np.round(_iou, 2)) + ".jpg"))
        tbar.close()


if __name__ == '__main__':
    for model_name in model_list:
        for loss_name in loss_list:
            print('#---------------------------------------------------------------')
            print('model: ', model_name)
            print('loss: ', loss_name)
            model = Trainer(model_name=model_name, loss_name=loss_name)
            if model.current_epoch <= total_epoch:
                model.run()
            print('#---------------------------------------------------------------\n')
