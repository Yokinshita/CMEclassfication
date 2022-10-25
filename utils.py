from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from natsort import natsorted
from typing import Union, List, Callable, Optional
import gc
import cv2


def loadSingleImg(path: str, transform: Optional[Callable]) -> np.ndarray:
    """
    载入单张图片，形状为NCHW
    """
    img = Image.open(path).convert('L')
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    if transform:
        img = transform(img)
    # img = torch.from_numpy(img)
    return img


def loadImageFolder(
    path_to_folder: str,
    transform: Optional[Callable] = None,
    ordering: bool = True,
) -> np.ndarray:
    """
    载入给定文件夹中的所有图片，形状为NCHW

    返回np.ndarray
    """
    pics = os.listdir(path_to_folder)
    # 首先载入第一张图片
    if ordering is True:
        pics = natsorted(pics)  #对文件按照名称排序
    imgs = loadSingleImg(os.path.join(path_to_folder, pics[0]), transform)
    for i in range(1, len(pics)):
        img = loadSingleImg(os.path.join(path_to_folder, pics[i]), transform)
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


def drawImageArrays(*arrays, title: List[str] = None):
    '''绘制多个图像数组，这些图像数组应当具有同样的形状，为NHWC
    
    Parameters:
    -----------
    array : 待绘制的图像数组，形状为NHWC。
    title : 对每一个图像数组的标注，为str
    '''
    column = len(arrays)
    row = arrays[0].shape[0]
    for array in arrays:
        if row != array.shape[0]:
            raise ValueError(
                'First dimension of all array expected to be the same. Expected {} got {}'
                .format(row, array.shape[0]))
    if title is not None:
        if len(title) != column:
            raise ValueError(
                'Each array should have a title , got {} titles but have {} arrays'
                .format(len(title), column))
    plt.figure(figsize=(4 * column, 4 * row))
    for j in range(column):
        for i in range(row):
            plt.subplot(row, column, i * column + j + 1)
            if title is not None:
                plt.title('{} {}'.format(title[j], i),
                          fontsize=10,
                          color='black')
            else:
                plt.title(str(i), fontsize=10, color='black')
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
    return normoalizeArray(arr.astype(np.uint8), 0, 255)


def showImg(arr: np.ndarray):
    '''将数组作为图片展示

    Parameters
    ----------
    arr : np.ndarray
        需要展示的数组
    '''
    Image.fromarray(arrayToImg(arr)).show()


def grayImageToRGB(arr: np.ndarray) -> np.ndarray:
    '''将单通道图像的通道复制三份，成为RGB图像

    Parameters
    ----------
    arr : np.ndarray
        单通道图像，形状为HW或者NHW

    Returns
    -------
    np.ndarray
        RGB三通道图像，形状为HWC或者NHWC

    Raises
    ------
    ValueError
        图像的维度必须为2或者3
    '''
    if arr.ndim == 3:
        return np.concatenate((np.expand_dims(arr, 3), np.expand_dims(
            arr, 3), np.expand_dims(arr, 3)),
                              axis=3)
    elif arr.ndim == 2:
        return np.concatenate((np.expand_dims(arr, 2), np.expand_dims(
            arr, 2), np.expand_dims(arr, 2)),
                              axis=2)
    else:
        raise ValueError(
            'Dimensions of input array must be 3(NHW) or 2(HW),got {}'.format(
                arr.ndim))


def applyMaskOnImage(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    '''将遮罩应用到图片数组上，并返回应用mask后的图片数组

    Parameters
    ----------
    img : np.ndarray
        图片数组，形状为NHW或NHWC
    mask : np.ndarray
        图片遮罩，值为0/1，形状为NHW，为uint8

    Returns
    -------
    np.ndarray
        遮罩后图片数组，形状为NHW或NHWC，类型与输入数组相同
    '''
    maskedArray = np.zeros_like(img, dtype=img.dtype)
    for i in range(img.shape[0]):
        maskedArray[i] = cv2.bitwise_and(img[i], img[i], mask=mask[i])
    return maskedArray


class CenterCrop:
    """
    将CME图像中以circlePoint为圆心，半径为90的区域内像素置为需要的值
    """
    def __init__(self, fmat, circlePoint=(243, 258), radius=90, value=0):
        """

        Parameters
        ----------
        fmt : str
            输入图像的格式，CHW或者是NCHW。
            若为"CHW"，则图像通道数可以为1或者为3，若为"NCHW"，图像通道数应为3
        circlePoint : tuple, optional
            圆心点, by default (243, 258)
        radius : int, optional
            半径, by default 90
        value : float, optional
            要赋的值, 应当在[0,1]之内, by default 0
        """
        self.value = value
        self.fmat = fmat
        if fmat == 'CHW':
            self.mask = np.full((512, 512), False, dtype=bool)
            for i in range(512):
                for j in range(512):
                    if ((i - circlePoint[0])**2 +
                        (j - circlePoint[1])**2)**0.5 < radius:
                        self.mask[i, j] = True
        elif fmat == 'NCHW':
            self.mask = np.full((1, 512, 512), False, dtype=bool)
            for i in range(512):
                for j in range(512):
                    if ((i - circlePoint[0])**2 +
                        (j - circlePoint[1])**2)**0.5 < radius:
                        self.mask[:, i, j] = True
        else:
            raise ValueError('Input parameter format only accept CHW or NCHW')

    def __call__(self, image: Union[np.ndarray, torch.Tensor]):
        if isinstance(image, np.ndarray):
            for i in range(image.shape[0]):
                image[i][self.mask] = self.value
            return image
        elif isinstance(image, torch.Tensor):
            tensormask = torch.from_numpy(self.mask)
            for i in range(image.shape[0]):
                image[i][tensormask] = self.value
            return image
        else:
            raise ValueError(
                'Input array must be np.ndarray or torch.Tensor, got {} '.
                format(image.__class__.__name__))

    def __repr__(self):
        string = self.__class__.__name__ + '(fmat:{}, set center value to {})'.format(
            self.fmat, self.value)
        return string


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


class MemCache:
    '''PyTorch显存管理类
    详情可参见https://www.zhihu.com/question/274635237/answer/2433849613
    '''
    @staticmethod
    def byte2MB(bt):
        return round(bt / (1024**2), 3)

    def __init__(self):
        self.dctn = {}
        self.max_reserved = 0
        self.max_allocate = 0

    def mclean(self):
        r0 = torch.cuda.memory_reserved(0)
        a0 = torch.cuda.memory_allocated(0)
        f0 = r0 - a0

        for key in list(self.dctn.keys()):
            del self.dctn[key]
        gc.collect()
        torch.cuda.empty_cache()

        r1 = torch.cuda.memory_reserved(0)
        a1 = torch.cuda.memory_allocated(0)
        f1 = r1 - a1

        print('Mem Free')
        print(f'Reserved  \t {MemCache.byte2MB(r1 - r0)}MB')
        print(f'Allocated \t {MemCache.byte2MB(a1 - a0)}MB')
        print(f'Free      \t {MemCache.byte2MB(f1 - f0)}MB')

    def __setitem__(self, key, value):
        self.dctn[key] = value
        self.max_reserved = max(self.max_reserved,
                                torch.cuda.memory_reserved(0))
        self.max_allocate = max(self.max_allocate,
                                torch.cuda.memory_allocated(0))

    def __getitem__(self, item):
        return self.dctn[item]

    def __delitem__(self, *keys):
        r0 = torch.cuda.memory_reserved(0)
        a0 = torch.cuda.memory_allocated(0)
        f0 = r0 - a0

        for key in keys:
            del self.dctn[key]

        r1 = torch.cuda.memory_reserved(0)
        a1 = torch.cuda.memory_allocated(0)
        f1 = r1 - a1

        print('Cuda Free')
        print(f'Reserved  \t {MemCache.byte2MB(r1 - r0)}MB')
        print(f'Allocated \t {MemCache.byte2MB(a1 - a0)}MB')
        print(f'Free      \t {MemCache.byte2MB(f1 - f0)}MB')

    def show_cuda_info(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a

        print('Cuda Info')
        print(f'Total     \t{MemCache.byte2MB(t)} MB')
        print(
            f'Reserved  \t{MemCache.byte2MB(r)} [{MemCache.byte2MB(torch.cuda.max_memory_reserved(0))}] MB'
        )
        print(
            f'Allocated \t{MemCache.byte2MB(a)} [{MemCache.byte2MB(torch.cuda.max_memory_allocated(0))}] MB'
        )
        print(f'Free      \t{MemCache.byte2MB(f)} MB')


def allocatedMemoryOf(x: Union[torch.Tensor, torch.nn.Module]):
    '''输出Tensor或者Module所占的内存大小(以MB为单位)

    Parameters
    ----------
    x : Union[torch.Tensor, torch.nn.Module]
        需要查看大小的Tensor或者Module
    '''
    if isinstance(x, torch.Tensor):
        print(x.nelement() * x.element_size() / 1024 / 1024, 'MB')
    elif isinstance(x, torch.nn.Module):
        memAllocatedBytes = sum(p.nelement() * p.element_size()
                                for p in x.parameters())
        print(memAllocatedBytes / 1024 / 1024, 'MB')


class ToTensorNoDiv255:
    '''将PIL.Image转化为torch.Tensor，形状为CHW
    该类不会像torchvision.transform.ToTensor那样将数值除以255
    '''
    def __call__(self, pic):
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
        This function does not support torchscript.

        See :class:`~torchvision.transforms.ToTensor` for more details.

        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """

        default_float_dtype = torch.get_default_dtype()

        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.to(dtype=default_float_dtype)
            else:
                return img

        # handle PIL Image
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img = torch.from_numpy(
            np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

        if pic.mode == "1":
            img = 255 * img
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        # *在有无div(255)时参数可以更新，accu,loss均会变化
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
