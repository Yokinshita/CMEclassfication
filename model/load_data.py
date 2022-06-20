import torch
import torchvision
from PIL import Image
import numpy as np
import os
from typing import Union


def read_imgarray_from_singlepic(path_to_pic: str, transform):
    """
    读取图片，并返回numpy ndarray数组
    Arguments:
    ---------
    path_to_pic : 图片的路径

    Returns:
    -------
    img         : 图片数组，形状为1*高*宽
    """

    # png图片是P模式，要转换为L模式（灰度）
    img = Image.open(path_to_pic).convert('L')
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, 0)  # 增加一个维度，变成通道*高*宽的形式
    if transform:
        img = transform(img)
    return img


def read_imgarray_from_folder(path_to_folder: str, transform):
    """
    从单个文件夹中读取所有的图片，并转换为numpy.array形式返回
    Arguments:
    ---------
    path_to_folder   : 数据所属的根文件夹路径
    Returns:
    -------
    imgarray         : 某个文件夹下所有图片组成的数组，形状为数量*通道*高*宽
    """

    pics = os.listdir(path_to_folder)
    imglist = []
    for file in pics:
        img = read_imgarray_from_singlepic(os.path.join(path_to_folder, file),
                                           transform)
        imglist.append(img)
    imgarray = np.array(imglist)
    return imgarray


def load_CME(save_location, selected_remarks, transform):
    """
    将CME中selected_labels文件夹中的图片全部读取为imgarray
    Arguments:
    ---------
    path_to_folder   : 数据所属的根文件夹路径
    selected_labels  : 需要的数据所属的标签
    Returns:
    -------                                   
    imgarrays         : 所有图片组成的数组，形状为数量*通道*高*宽
    labels            : 数据的标签，1表示CME，0表示非CME
    """
    CME_path = os.path.join(save_location, 'CME')
    imgarray = read_imgarray_from_folder(
        os.path.join(CME_path, selected_remarks.pop(0)), transform)
    for remark in selected_remarks:
        current_label_imgarray = read_imgarray_from_folder(
            os.path.join(CME_path, remark), transform)
        print('Reading CME data from {}'.format(os.path.join(CME_path,
                                                             remark)))
        imgarray = np.concatenate((imgarray, current_label_imgarray), axis=0)
    labels = np.ones(imgarray.shape[0], dtype=np.int64)
    return imgarray, labels


def load_no_CME(save_location, transform):
    """
    加载非CME图片数据
    Arguments:
    ---------
    path_to_folder   : 数据所属的根文件夹路径
    Returns:
    -------
    imgarrays         : 所有图片组成的数组，形状为数量*高*宽*通道
    labels            : 数据的标签，1表示CME，0表示非CME
    """

    No_CME_path = os.path.join(save_location, 'No CME')
    print('Reading No CME data from {}'.format(No_CME_path))
    imgarray = read_imgarray_from_folder(No_CME_path, transform)
    labels = np.zeros(imgarray.shape[0], dtype=np.int64)
    return imgarray, labels


class CenterCrop:
    """
    将CME图像中以circlePoint为圆心，半径为90的区域内像素置为需要的值
    为提高效率，利用image*mask+value的方式对中心区域进行赋值
    """

    def __init__(self, circlePoint=(243, 258), radius=90, value=127):
        """

        Parameters
        ----------
        circlePoint : tuple, optional
            圆心点, by default (243, 258)
        radius : int, optional
            半径, by default 90
        value : int, optional
            要赋的值, by default 127
        """
        self.mask = np.ones((512, 512))
        self.value = np.zeros((512, 512))
        for i in range(512):
            for j in range(512):
                if ((i - circlePoint[0])**2 +
                    (j - circlePoint[1])**2)**0.5 < radius:
                    self.mask[i, j] = 0
                    self.value[i, j] = value

    def __call__(self, image: Union[np.ndarray, torch.Tensor]):
        if isinstance(image, np.ndarray):
            image = np.copy(image)
            if image.ndim == 3:
                image[0] = np.multiply(image[0], self.mask) + self.value
                return image
            elif image.ndim == 4:
                for i in range(image.shape[0]):
                    image[i][0] = np.multiply(image[i][0],
                                              self.mask) + self.value
                return image
        if isinstance(image, torch.Tensor):
            mask = torch.from_numpy(self.mask)
            mask = torch.clone(mask)
            if image.ndim == 3:
                image[0] = torch.mul(image[0], mask) + self.value
                return image
            elif image.ndim == 4:
                for i in range(image.shape[0]):
                    image[i][0] = torch.mul(image[i][0], mask) + self.value
                return image


class CMEdata:
    # 该类用以载入数据集，同时会将CME和非CME数据混合后打乱，并可以以TensorDataset形式输出
    def __init__(self, save_location: str, selected_remarks: list,
                 train_percentage):
        """

        Arguments:
        ---------
        save_location       : 数据的根目录
        selected_remarks    : 需要用做数据集的图片所属的标签
        train_percentage    : 训练集的占比

        """

        self.save_location = save_location
        self.selected_remarks = selected_remarks
        self.train_percentage = train_percentage
        self.trans = CenterCrop()

    def __random_split(self, data: np.ndarray, labels: np.ndarray,
                       train_percentage: float):
        size = data.shape[0]  # 数据集中数据的个数
        index = np.arange(size)  # 产生索引
        np.random.shuffle(index)  # 打乱索引，以便将数据都混合在一起
        split = int(train_percentage * size)  # 获得训练集和测试集的分划点
        # 0到split为训练集 split到最后为测试集
        train_index, test_index = index[:split], index[split:]
        train_data = data[train_index, :, :, :]
        train_label = labels[train_index]
        test_data = data[test_index, :, :, :]
        test_label = labels[test_index]
        return train_data, train_label, test_data, test_label

    def save_data_to_npz(self):
        # 储存数据集和标签
        npz_save_path = os.path.join(self.save_location, 'npz')
        print('npz data save to {}'.format(npz_save_path))
        if not os.path.exists(npz_save_path):
            os.makedirs(npz_save_path)
        np.savez(os.path.join(npz_save_path, 'data.npz'),
                 train_data=self.train_data,
                 train_label=self.train_label,
                 test_data=self.test_data,
                 test_label=self.test_label)

    def load_data_from_pic(self, train_percentage):
        """
        加载图片数据，打乱，分割并作为数据集和测试集
        ！！！！使用该方法，每次得到的训练集和测试集不完全相同
        Arguments:
        ---------
        train_percentage : 训练集所占全部数据集的比重，使用该方法时，需要自行划分训练集与测试集
        """
        print('Loading data from {}'.format(self.save_location))
        CMEdata, CME_labels = load_CME(self.save_location,
                                       self.selected_remarks,
                                       transform=self.trans)
        no_CME_data, no_CME_labels = load_no_CME(self.save_location,
                                                 transform=self.trans)
        data = np.concatenate((CMEdata, no_CME_data), axis=0)
        labels = np.concatenate((CME_labels, no_CME_labels), axis=0)
        self.size = data.shape[0]
        self.train_size = int(self.size * train_percentage)
        self.test_size = self.size - self.train_size
        self.train_data, self.train_label, self.test_data, self.test_label = self.__random_split(
            data, labels, train_percentage)
        npz_file_path = os.path.join(self.save_location, 'npz', 'data.npz')
        if not os.path.exists(npz_file_path):
            self.save_data_to_npz()

    def load_data_from_npz(self):
        # 从已经储存的npz文件中加载数据
        # ！！！使用此方法可以保证得到的数据集和测试集均相同
        npz_save_path = os.path.join(self.save_location, 'npz')
        print('Loading data from {}'.format(npz_save_path))
        data = np.load(os.path.join(npz_save_path, 'data.npz'))
        self.train_data = data['train_data']
        self.train_label = data['train_label']
        self.test_data = data['test_data']
        self.test_label = data['test_label']
        self.size = self.train_data.shape[0] + self.test_data.shape[0]
        self.train_size = self.train_data.shape[0]
        self.test_size = self.size - self.train_size

    def load_data(self, forcing_load_from_pic=False):
        npz_file_path = os.path.join(self.save_location, 'npz', 'data.npz')
        # 存在npz文件并且不强制从图片载入，则从npz文件载入
        if os.path.exists(npz_file_path) and forcing_load_from_pic is False:
            print('Pic npz file exists in {},load data from npz file'.format(
                npz_file_path))
            self.load_data_from_npz()
        else:
            self.load_data_from_pic(self.train_percentage)

    def to_tensordataset(self, is_train=True):
        """

        Arguments:
        ---------
        is_train      : 若为True，输出训练集数据，若为False输出测试集数据

        Returns:
        -------
        TensorDataset : 返回训练集或者测试集数据，可用于初始化DataLoader
        """
        if is_train:
            feature = self.train_data
            label = self.train_label
        else:
            feature = self.test_data
            label = self.test_label
        # 转换为tensor
        feature = torch.from_numpy(feature)
        label = torch.from_numpy(label)

        return torch.utils.data.TensorDataset(feature, label)


if __name__ == '__main__':
    # 该文件被设置为可以独立运行
    save_location = r'D:\Programming\CME_data'
    selected_remarks = ['Halo', 'No Remark', 'Partial Halo']
    train_percentage = 0.7
    batch_size = 100
    cmedata = CMEdata(save_location, selected_remarks, train_percentage)
    cmedata.load_data_from_pic(train_percentage)
    cmedata.save_data_to_npz()
    train_dataset = cmedata.to_tensordataset()
    train_iter = torch.utils.data.DataLoader(train_dataset,
                                             batch_size,
                                             shuffle=True)
    for X, y in train_iter:
        print(X.shape)
        print(y.shape)
    print('yes')
    print('finished')
