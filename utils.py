from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from natsort import natsorted
from typing import Union, List, Callable, Optional, Tuple
import gc
import cv2
from datetime import datetime


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


def loadImageFolder(path_to_folder: str,
                    transform: Optional[Callable] = None,
                    ordering: bool = True,
                    regex: Optional[str] = None) -> np.ndarray:
    '''载入给定文件夹内文件名匹配regex成功的图片，并返回一个形状为NCHW的数组

    Parameters
    ----------
    path_to_folder : str
        文件夹路径
    transform : Optional[Callable], optional
        对图片进行的变换, by default None
    ordering : bool, optional
        若为True,则将对文件名进行自然排序后载入, by default True
    regex : Optional[str], optional
        文件名应当符合的正则表达式,若为None,则载入所有图片, by default None

    Returns
    -------
    np.ndarray
        文件夹图片组成得到的数组，形状为NCHW
    '''
    import re

    pics = os.listdir(path_to_folder)
    if ordering:
        pics = natsorted(pics)  # 对文件按照名称排序
    imgs = []
    for pic in pics:
        if regex:
            if re.search(regex, pic):
                img = loadSingleImg(os.path.join(path_to_folder, pic),
                                    transform)
                imgs.append(img)
        else:
            img = loadSingleImg(os.path.join(path_to_folder, pic), transform)
            imgs.append(img)
    imgs = np.concatenate(imgs, axis=0)
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


def drawImageArrays(*arrays, title: Optional[List[str]] = None):
    '''绘制多个图像数组，这些图像数组的第一维应当相同，它们的形状为NHWC或NHW

    Parameters:
    -----------
    array : 待绘制的图像数组，形状为NHWC或NHW。
    title : 对每一个图像数组的标注，为str
    '''
    column = len(arrays)
    row = arrays[0].shape[0]
    for array in arrays:
        if row != array.shape[0]:
            raise ValueError(
                'First dimension of all array expected to be the same. Expected {} got {}'
                .format(row, array.shape[0]))
        if array.ndim != 3 and array.ndim != 4:
            raise ValueError(
                'Array expected to be 3 or 4 dimensional, got {} dimensional array'
                .format(array.ndim))
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
            if arrays[j][i].ndim == 3 and arrays[j][i].shape[2] == 3:
                plt.imshow(arrays[j][i])  # 图片形状是NHWC(RGB)的情况
            else:
                plt.imshow(arrays[j][i], cmap='gray')  # 图片是灰度图的情况
            plt.xticks(())
            plt.yticks(())
    plt.show()


def drawImageArrayInFlat(image: np.ndarray, cols=5) -> None:
    '''以平铺的布局绘制image数组

    Parameters
    ----------
    image : np.ndarray
        需要绘制的数组，形状为NHW或NHWC
    cols : int, optional
        每一行绘制的图片数, by default 5
    '''
    from math import ceil
    nums = len(image)
    rows = ceil(nums / cols)
    plt.figure(figsize=(cols * 5, rows * 5))
    ind = 0
    for i in range(rows):
        for j in range(cols):
            ax = plt.subplot(rows, cols, ind + 1)
            if image.ndim == 4 and image.shape[3] == 3:
                ax.imshow(image[ind])
            else:
                ax.imshow(image[ind], cmap='gray')
            ax.set_title(str(ind))
            ind += 1
            if ind >= nums:
                break
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
        self.radius = radius
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
        string = self.__class__.__name__ + '(fmat:{}, Radius={}, CenterValue={})'.format(
            self.fmat, self.radius, self.value)
        return string


class PylonCrop():
    def __init__(
            self,
            angle: int,
            width: Union[Tuple[int, int], int],
            center: Tuple[int, int] = (256, 242),
    ):
        '''遮挡C3日冕仪图像上的挡杆，返回遮挡处理后的图像

        Parameters
        ----------
        center : Tuple[int, int], default (256, 244)
            图像中心点坐标[x,y]
        angle : int
            挡杆位置的中心角度
        width : Union[Tuple[int, int], int]
            遮挡范围的宽度，当为整数时，遮挡范围为[angle-width, angle+width]，
            当为元组时，遮挡范围为[angle-width[0], angle+width[1]]
        '''
        self.center = center
        self.angle = angle
        self.width = width

    @staticmethod
    def atan2(dx, dy):
        from math import degrees, atan
        # 图像的坐标系的x,y轴正方向分别向右、向下，与一般坐标系不同，dx,dy的正负需要考虑此不同点
        if dx <= 0 and dy < 0:
            return degrees(atan(dx / dy))
        elif dx < 0 and dy == 0:
            return 90
        elif dx < 0 and dy > 0:
            return degrees(atan(dy / -dx)) + 90
        elif dx == 0 and dy > 0:
            return 180
        elif dx > 0 and dy > 0:
            return degrees(atan(dx / dy)) + 180
        elif dx > 0 and dy == 0:
            return 270
        elif dx > 0 and dy < 0:
            return degrees(atan(-dy / dx)) + 270
        else:
            return 0

    def __call__(self, img: np.ndarray):
        mask = np.ones_like(img)
        if type(self.width) == int:
            angleRange = [self.angle - self.width, self.angle + self.width]
        elif type(self.width) == tuple:
            angleRange = [
                self.angle - self.width[0], self.angle + self.width[1]
            ]
        else:
            raise TypeError("width must be int or tuple")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dx = j - self.center[0]
                dy = i - self.center[1]
                degree = self.atan2(dx, dy)
                if angleRange[0] <= degree <= angleRange[1]:
                    mask[i, j] = 0
        return img * mask


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


def downloadLascoImageBetween(start: datetime,
                              end: datetime,
                              imagePath: str = '/',
                              imageType: str = 'both'):
    '''下载给定起始时间段内的日冕仪图像

    note: datetime.strptime函数的格式化形式类似%Y%m%d_%H%M%S

    Parameters
    ----------
    start : datetime
        起始时间
    end : datetime
        结束时间
    picPath : str, optional
        图片文件保存目录, by default 'pic'
    picType : str, optinal
        下载图片的类型，可为C2，C3或both
    '''
    import requests
    from urllib import parse
    import re

    if imageType.lower() == 'c2':
        picHrefRegex = re.compile(r'<a href="(\d{8}_\d{6}_lasc2rdf.png)">')
    elif imageType.lower() == 'c3':
        picHrefRegex = re.compile(r'<a href="(\d{8}_\d{6}_lasc3rdf.png)">')
    elif imageType.lower() == 'both':
        picHrefRegex = re.compile(r'<a href="(\d{8}_\d{6}_lasc\drdf.png)">')
    else:
        raise ValueError(
            'Argument picType must be either c2, c3 or both, got {}'.format(
                imageType))
    if not os.path.exists(imagePath):
        os.mkdir(imagePath)
    CMEdailyPicsURL = 'https://cdaw.gsfc.nasa.gov/images/soho/lasco/{0:}/{1:0>2d}/{2:0>2d}/'.format(
        start.year, start.month, start.day)
    dailyPicPage = requests.get(CMEdailyPicsURL)
    hrefs = picHrefRegex.findall(dailyPicPage.text)  # 包含图片文件名的列表
    picDatetimeRegex = re.compile(r'(\d{8}_\d{6})_')  # 用来识别href中图片时间的正则表达式
    for i, href in enumerate(hrefs):
        picTime = picDatetimeRegex.findall(hrefs[i])[0]
        picTime = datetime.strptime(picTime, '%Y%m%d_%H%M%S')
        if start <= picTime <= end:
            picURL = parse.urljoin(CMEdailyPicsURL, href)
            try:
                res = requests.get(picURL)
            except Exception:
                print('Error:downloading pic {} fail'.format(href))
                continue
            with open(os.path.join(imagePath, href), 'wb') as f:
                f.write(res.content)


def downloadLascoImageAt(imageTime: datetime,
                         folderPath: str = '/',
                         imageType: Optional[str] = None) -> bool:
    '''下载某一时刻的Lasco日冕仪差分图像

    Parameters
    ----------
    imageTime : datetime.datetime
        日冕仪差分图像的时刻,
        datetime.strptime函数的格式化形式类似%Y/%m/%d %H:%M:%S
    folderPath : str, optional
        下载图片存放文件夹, by default '/'
    imageType : Optional[str], optional
        差分图像类型, by default None

    Returns
    -------
    bool
        下载若成功则返回True
    '''
    from urllib import parse
    import requests

    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    if imageType:
        imageType = imageType.lower()
    picFileNameC2 = '{}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}_lasc2rdf.png'.format(
        imageTime.year, imageTime.month, imageTime.day, imageTime.hour,
        imageTime.minute, imageTime.second)
    picFileNameC3 = '{}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}_lasc3rdf.png'.format(
        imageTime.year, imageTime.month, imageTime.day, imageTime.hour,
        imageTime.minute, imageTime.second)
    CMEdailyImageURL = 'https://cdaw.gsfc.nasa.gov/images/soho/lasco/{0:}/{1:0>2d}/{2:0>2d}/'.format(
        imageTime.year, imageTime.month, imageTime.day)
    imageURLC2 = parse.urljoin(CMEdailyImageURL, picFileNameC2)
    imageURLC3 = parse.urljoin(CMEdailyImageURL, picFileNameC3)
    resC2 = requests.get(imageURLC2)
    resC3 = requests.get(imageURLC3)
    if resC2.status_code == 200:
        res = resC2
        picFileName = picFileNameC2
    elif resC3.status_code == 200:
        res = resC3
        picFileName = picFileNameC3
    else:
        return False
    with open(os.path.join(folderPath, picFileName), 'wb') as f:
        f.write(res.content)
    return True


def downloadLascoImageFromDetailedInfo(url: str,
                                       folderPath: str = '/') -> None:
    '''从CME日志详细记录页中下载对应时间的CME差分图片

    Parameters
    ----------
    url : str
        CME日志详细记录页的URL
    folderPath : str, optional
        下载图片存放文件夹路径, by default '/'
    '''

    import requests
    import re

    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    res = requests.get(url)
    regex = re.compile(r'(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}).*C(\d)')
    recordBeginRow = None
    for i, row in enumerate(res.text.split('\n')):
        if row.startswith('# HEIGHT'):
            recordBeginRow = i
        if recordBeginRow and i > recordBeginRow:
            regexSearch = regex.findall(row)
            if regexSearch:
                imageTime = regexSearch[0][0]
                imageType = 'c' + regexSearch[0][1]
                imageTime = datetime.strptime(imageTime, '%Y/%m/%d %H:%M:%S')
                downloadLascoImageAt(imageTime, folderPath, imageType)
