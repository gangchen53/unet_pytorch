from PIL import Image
from torch.utils.data import Dataset
import utils.seg_transforms as seg_tr


def read_txt(txt_path):
    file_path = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_path.append(line.strip())
    return file_path


def save_txt(txt_path, content):
    with open(txt_path, 'w') as f:
        for line in content:
            f.write(line)
            f.write('\n')


class ImageDataset(Dataset):
    def __init__(self, image_path, label_path, flag='train'):
        self.image_path = image_path
        self.label_path = label_path
        self.flag = flag

        self.train_transforms = seg_tr.Compose([
            seg_tr.RandomCrop(256),
            seg_tr.RandomRotation(degrees=15),
            seg_tr.RandomHorizontalFlip(flip_prob=0.5),
            seg_tr.ToTensor()
        ])

        self.val_transforms = seg_tr.Compose([
            seg_tr.CenterCrop(512),
            seg_tr.ToTensor()
        ])

        self.test_transforms = seg_tr.Compose([
            seg_tr.CenterCrop(512),
            seg_tr.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert('RGB')
        label = Image.open(self.label_path[idx]).convert('L')
        if self.flag == 'train':
            return self.train_transforms(image, label)
        elif self.flag == 'val':
            return self.val_transforms(image, label)
        elif self.flag == 'test':
            return self.test_transforms(image, label)
        else:
            raise NotImplementedError
