import os
import random
from pathlib import Path


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


if __name__ == '__main__':
    train_images_path_list = list()
    train_labels_path_list = list()

    val_images_path_list = list()
    val_labels_path_list = list()

    test_images_path_list = list()
    test_labels_path_list = list()

    root_dir = Path('datasets')
    with os.scandir(str(root_dir)) as it:
        for entry in it:
            if entry.is_dir():
                print(entry.name)
                if 'GroundTruth' in entry.name:
                    if 'Test' in entry.name:
                        test_labels_dir = Path(entry.path)
                    else:
                        train_labels_dir = Path(entry.path)

        train_labels_path = list(train_labels_dir.glob('*.png'))
        random.shuffle(train_labels_path)
        for i, train_lbl_path in enumerate(train_labels_path):
            train_img_path = Path(
                str(train_lbl_path).rsplit('_', maxsplit=1)[0].replace('GroundTruth', 'Data')
            ).with_suffix('.jpg')
            if i <= 799:
                train_images_path_list.append(str(train_img_path))
                train_labels_path_list.append(str(train_lbl_path))
            else:
                val_images_path_list.append(str(train_img_path))
                val_labels_path_list.append(str(train_lbl_path))

        test_labels_path = list(test_labels_dir.glob('*.png'))
        for test_lbl_path in test_labels_path:
            test_img_path = Path(
                str(test_lbl_path).rsplit('_', maxsplit=1)[0].replace('GroundTruth', 'Data')
            ).with_suffix('.jpg')
            test_images_path_list.append(str(test_img_path))
            test_labels_path_list.append(str(test_lbl_path))

        save_txt(txt_path='datasets/images_train.txt', content=train_images_path_list)
        save_txt(txt_path='datasets/labels_train.txt', content=train_labels_path_list)

        save_txt(txt_path='datasets/images_val.txt', content=val_images_path_list)
        save_txt(txt_path='datasets/labels_val.txt', content=val_labels_path_list)

        save_txt(txt_path='datasets/images_test.txt', content=test_images_path_list)
        save_txt(txt_path='datasets/labels_test.txt', content=test_labels_path_list)
