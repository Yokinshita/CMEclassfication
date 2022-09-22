from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from natsort import natsorted
from typing import Union


def loadSingleImg(path: str) -> np.ndarray:
    """
    载入单张图片，形状为NCHW
    """
    img = Image.open(path).convert('L')
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    # img = torch.from_numpy(img)
    return img


def loadImageFolder(path_to_folder: str, ordering: bool = True) -> np.ndarray:
    """
    载入给定文件夹中的所有图片，形状为NCHW

    返回np.ndarray
    """
    pics = os.listdir(path_to_folder)
    # 首先载入第一张图片
    if ordering is True:
        pics = natsorted(pics)  #对文件按照名称排序
    imgs = loadSingleImg(os.path.join(path_to_folder, pics[0]))
    for i in range(1, len(pics)):
        img = loadSingleImg(os.path.join(path_to_folder, pics[i]))
        imgs = np.concatenate((imgs, img), axis=0)
    return imgs


def arrayToPic(array: np.ndarray) -> Image.Image:
    '''将仅包含[0,1]值的二元数组转化为PIL.Image图片类

    Parameters
    ----------
    array : np.ndarray
        待转化的[0,1]值的数组

    Returns
    -------
    Image.Image
        PIL.Image类的图片
    '''
    return Image.fromarray(255 * array.astype('int')).convert('L')


def drawImageArray(img: np.ndarray):
    for i in range(img.shape[0]):
        plt.figure(figsize=(7.07, 7.07))
        plt.imshow(img[i], cmap='gray')
        plt.xticks(())
        plt.yticks(())


def savefig(array: np.ndarray, path: str, preffix='pic'):
    """保存数组为图片

    Parameters
    ----------
    array : np.ndarray
        需要保存的数组，形状为NHW
    path : str
        路径
    preffix : str
        图片文件名称的前缀
    """
    for i in range(array.shape[0]):
        Image.fromarray(array[i].astype('uint8')).convert('L').save(
            os.path.join(path, preffix + '{}.png'.format(i)))


def drawImageArrays(*arrays):
    '''绘制多个图像数组，这些图像数组应当具有同样的形状，为NHWC
    
    Parameters:
    -----------
    array : 待绘制的图像数组，形状为NHWC。
    '''
    column = len(arrays)
    row = arrays[0].shape[0]
    for array in arrays:
        if row != array.shape[0]:
            raise ValueError(
                'First dimension of all array expected to be the same. Expected {} got {}'
                .format(row, array.shape[0]))
    plt.figure(figsize=(4 * column, 4 * row))
    for j in range(column):
        for i in range(row):
            plt.subplot(row, column, i * column + j + 1)
            plt.title(str(i), fontsize=10, color='white')
            if arrays[j][i].shape[2] == 1:
                plt.imshow(arrays[j][i], cmap='gray')  # 图片是灰度图的情况
            elif arrays[j][i].shape[2] == 3:
                plt.imshow(arrays[j][i])  # 图片是RGB的情况
            plt.xticks(())
            plt.yticks(())
    plt.show()


def normoalizeArray(array: np.ndarray, newmin, newmax) -> np.ndarray:
    '''
    将array中的最大值变为max，最小值变为min，其他的数值按照平均原则计算。
    Parameters
    ----------
    array:需要改变的的数组
    newmin:最小值
    newmax:最大值

    Returns
    -------
    值改变后的数组
    '''
    originMax = array.max()
    originMin = array.min()
    difference = originMax - originMin
    newArray = (array - originMin) / difference * (newmax - newmin) + newmin
    return newArray


def arrayToImg(arr: np.ndarray) -> np.ndarray:
    '''将数组的值改为[0,255]范围内

    Parameters
    ----------
    arr : np.ndarray
        被改变的数组

    Returns
    -------
    np.ndarray
        修改后的数组
    '''
    return normoalizeArray(arr.astype('np.uint8'), 0, 255)


def showImg(arr: np.ndarray):
    '''将数组作为图片展示

    Parameters
    ----------
    arr : np.ndarray
        需要展示的数组
    '''
    Image.fromarray(arrayToImg(arr)).show()


def grayImageToRGB(arr: np.ndarray) -> np.ndarray:
    '''将灰度图像的通道复制三份，成为RGB图像

    Parameters
    ----------
    arr : np.ndarray
        灰度图像，形状为HW或者NHW

    Returns
    -------
    np.ndarray
        RGB三通道图像，形状为HWC或者NHWC

    Raises
    ------
    ValueError
        图像的维度必须为2，否则会引发ValueError
    '''
    if arr.ndim == 3:
        return np.concatenate((np.expand_dims(arr, 3), np.expand_dims(
            arr, 3), np.expand_dims(arr, 3)),
                              axis=3).astype('uint8')
    elif arr.ndim == 2:
        return np.concatenate((np.expand_dims(arr, 2), np.expand_dims(
            arr, 2), np.expand_dims(arr, 2)),
                              axis=2).astype('uint8')
    else:
        raise ValueError(
            'Dimensions of input array must be 3(NHW) or 2(HW),got {}'.format(
                arr.ndim))


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
            else:
                raise ValueError(
                    'Input array dimensions expected to be 3 or 4, got {}'.
                    format(image.ndim))
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
            else:
                raise ValueError(
                    'Input array dimensions expected to be 3 or 4, got {}'.
                    format(image.ndim))


def showArrayRange(arr: np.ndarray) -> tuple:
    '''打印arr的最大值和最小值

    Parameters
    ----------
    arr : np.ndarray
        待打印的数组
    '''
    return arr.max(), arr.min()


def examinePerChannel(arr: np.ndarray) -> np.ndarray:
    '''检查图像数组的每个通道是否完全一致

    Parameters
    ----------
    arr : np.ndarray
        待检查的数组，形状为HWC

    Returns
    -------
    np.ndarray
        通道不一致的像素的坐标
    '''
    if arr.ndim == 3:
        differ01 = np.argwhere(arr[:, :][0] != arr[:, :][1])
        differ02 = np.argwhere(arr[:, :][0] != arr[:, :][2])
        differ = np.concatenate((differ01, differ02), axis=0)
        return differ


def NCHWtoNHWC(array: Union[np.ndarray, torch.Tensor]):
    '''将数组由NCHW转换为NHWC

    Parameters
    ----------
    array : Union[np.ndarray, torch.Tensor]
        待转换的数组

    Returns
    -------
    Union[np.ndarray,torch.Tensor]
        NHWC形状的数组
    '''
    if isinstance(array, np.ndarray):
        return array.transpose((0, 2, 3, 1))
    elif isinstance(array, torch.Tensor):
        return array.permute((0, 2, 3, 1))
    else:
        raise TypeError(
            'Input array type should be np.ndarray or torch.Tensor')


def NHWCtoNCHW(array: Union[np.ndarray, torch.Tensor]):
    '''将数组由NHWC转换为NCHW

    Parameters
    ----------
    array : Union[np.ndarray, torch.Tensor]
        待转换的数组

    Returns
    -------
    Union[np.ndarray,torch.Tensor]
        NCHW形状的数组
    '''
    if isinstance(array, np.ndarray):
        return array.transpose((0, 3, 1, 2))
    elif isinstance(array, torch.Tensor):
        return array.permute((0, 3, 1, 2))
    else:
        raise TypeError(
            'Input array type should be np.ndarray or torch.Tensor')
