from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import model.train_schedule
import model.model_defination
import torch
from scipy.interpolate import griddata
import os
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


def loadImages(path_to_folder: str) -> np.ndarray:
    """
    载入给定文件夹中的所有图片，形状为NCHW

    返回np.ndarray
    """
    pics = os.listdir(path_to_folder)
    # 首先载入第一张图片
    imgs = loadSingleImg(os.path.join(path_to_folder, pics[0]))
    for i in range(1, len(pics)):
        img = loadSingleImg(os.path.join(path_to_folder, pics[i]))
        imgs = np.concatenate((imgs, img), axis=0)
    return imgs


def arrayToPic(array: np.ndarray) -> Image.Image:
    """将数组转换为图片"""
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
    """
    for i in range(array.shape[0]):
        Image.fromarray(array[i].astype('int')).convert('L').save(
            os.path.join(path, preffix + '{}.png'.format(i)))


def drawImageArrays(*arrays):
    column = len(arrays)
    row = arrays[0].shape[0]
    for array in arrays:
        if row != array.shape[0]:
            raise ValueError(
                'Shape of all array expected to be the same.Expected {} got {}'
                .format(row, array.shape[0]))
        if array.ndim != 3:
            raise ValueError(
                'Array dimension expected to be 3 , got {}'.format(array.ndim))
    plt.figure(figsize=(3.6 * column, 4 * row))
    for i in range(row):
        for j in range(column):
            plt.subplot(row, column, i * column + j + 1)
            plt.title(str(i), fontsize=10, color='white')
            plt.imshow(arrays[j][i], cmap='gray')
            plt.xticks(())
            plt.yticks(())
    plt.show()


def getActivation(net: torch.nn.Module, x: Union[torch.Tensor,
                                                 np.ndarray]) -> np.ndarray:
    """ 
    获得最后一层卷积层的输出
    
    输出结果的维度为N*h*w*d
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    net.eval()
    activation = []

    def forward_hook(modeul, data_input, data_output):
        activation.append(data_output.detach().permute(0, 2, 3, 1).numpy())

    net.conv2.register_forward_hook(forward_hook)
    out = net(x)
    return activation[0]


def getMeanVector(x: np.ndarray):
    """
    获得所有descriptor的平均向量
    
    x为N*h*w*d维的np.array
    """
    return np.mean(x, axis=(0, 1, 2))


def cov(x: np.ndarray):
    """
    获得协方差矩阵
    
    x为N*h*w*d维的np.ndarray
    """
    k = x.shape[0] * x.shape[1] * x.shape[2]
    xMeanVector = getMeanVector(x)
    convMat = np.zeros(x.shape[3])
    for n in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                deviaVector = x[n][i][j] - xMeanVector
                # 对x中取出的descripter向量进行升维
                # 因为直接取出的descripter向量是一维的，直接相乘会出现问题，需转化为列向量
                deviaVector = np.expand_dims(deviaVector, axis=1)
                tempMat = np.matmul(deviaVector, deviaVector.T)
                convMat = convMat + tempMat
    return convMat / k


def getPrinCompVector(activation: np.ndarray) -> np.ndarray:
    """
    获得主成分向量
    
    activation的形状为N*h*w*d，是卷积层输出的特征图
    """
    covMatrix = cov(activation)
    eigValue, eigVector = np.linalg.eig(covMatrix)
    prinCompInd = np.argmax(eigValue)
    prinCompVector = eigVector[:, prinCompInd]
    # prinComp形状为(50,)，对其增加一维变为列向量
    prinCompVector = np.expand_dims(prinCompVector, axis=1)
    return prinCompVector


def getIndicatorMatrix(activation: np.ndarray, ind: int,
                       prinCompVector: np.ndarray) -> np.ndarray:
    """获得索引为ind的图片的activation所对应的Indicator Matrix

    Parameters
    ----------
    activation : np.ndarray
        维度为N*h*w*d,是N张图片的激活层构成的数组
    ind : int
        表示需求出指示矩阵的图片对应的索引
    prinCompVector : np.ndarray
        最大特征值对应的主成分向量

    Returns
    -------
    np.ndarray 
        指示矩阵
    """
    img = activation[ind]
    xMeanVector = getMeanVector(activation)
    indicatorMatrix = np.zeros((activation.shape[1], activation.shape[2]))
    for i in range(activation.shape[1]):
        for j in range(activation.shape[2]):
            indicatorMatrix[i, j] = np.matmul(prinCompVector.T,
                                              img[i, j] - xMeanVector)
    return indicatorMatrix


def reSize(x: np.ndarray, targetSize=(512, 512)) -> np.ndarray:
    """利用最近邻插值，更改矩阵大小

    Parameters
    ----------
    x : np.ndarray
        需要变换的矩阵
    targetSize : tuple, optional
        目标大小, by default (512, 512)

    Returns
    -------
    np.ndarray
        修改大小后的矩阵
    """
    pointXCoord = np.floor(np.linspace(0, targetSize[0] - 1, x.shape[0]))
    pointYCoord = np.floor(np.linspace(0, targetSize[1] - 1, x.shape[1]))
    pointCoord = np.array([(i, j) for i in pointXCoord for j in pointYCoord])
    X = np.arange(0, targetSize[0])
    Y = np.arange(0, targetSize[1])
    X, Y = np.meshgrid(X, Y)
    reSizedX = griddata(pointCoord, x.flatten(), (X, Y), method='nearest')
    # 此处返回的数组应为插值后的数组的转置，原因在于meshgrid生成的X,Y数组的顺序不同
    reSizedX = reSizedX.T
    return reSizedX


def getNextStartPoint(mask):
    '''查找下一个未被标记的点的坐标，若有这样的点则返回一个这样的点的坐标，若无则返回None'''
    ind = np.argwhere(mask == 0)
    if ind.size > 0:
        return tuple(ind[0])
    else:
        return None


def isInside(point: tuple, xBound: tuple, yBound: tuple) -> bool:
    '''
    判断点point是否在界限内
    
    xBound和yBound均为二元组，分别为x和y坐标的上下界。
    '''
    if xBound[0] <= point[0] <= xBound[1] and yBound[0] <= point[1] <= yBound[
            1]:
        return True
    else:
        return False


def getConnectedComponet(reSizedIndicator: np.ndarray) -> tuple:
    """找到指示矩阵中的最大连通分量

    Parameters
    ----------
    reSizedIndicator : np.ndarray
        指示矩阵 形状为h*w

    Returns
    -------
    mask : np.ndarray
        连通分量的遮罩,为h*w二维数组,mask中大于0的值表示连通分量,不同的大于0的值表示不同的连通分量。
    componentIndex : int
        表示连通分量的个数
    """
    if reSizedIndicator.ndim != 2:
        raise ValueError('Dimensions of input array must be 2.Got {}'.format(
            reSizedIndicator.ndim))
    binaryIndicatorMat = np.where(reSizedIndicator > 0, 1, 0)
    # mask用于指示reSizedIndicator中同位置的点是否被标记
    # 若某点为0，表示还未被搜索到，若为-1，表示此点不在搜索区域内，若为正数，则用以区分不同的连通分量
    mask = np.zeros_like(reSizedIndicator)
    # binaryIndicatorMat中为0的点不属于搜索范围，需要在fill中将相应的点标为-1
    mask[binaryIndicatorMat == 0] = -1
    # 指定flood-fill算法的起始点坐标
    # mask中起始点所对应的位置的值必须为0
    filled = set()
    #s = (0, 3)
    s = getNextStartPoint(mask)
    if s is None:
        print('没有可供选择的起始点，mask中所有点都被标记了')
    assert mask[s[0]][s[1]] == 0, '起始点不满足要求，请重新选择flood-fill算法起始点'
    fill = set()
    fill.add(s)
    height, width = reSizedIndicator.shape[0] - 1, reSizedIndicator.shape[1] - 1
    # componentIndex用于指示不同的连接分量，由1开始依次累加1
    componentIndex = 1
    while fill:
        r, c = fill.pop()
        # 去掉以下判断并在向fill中添加上下左右点时增加对界限的判断是因为
        # 当(r,c)位于边界处，且此时fill为空时
        # 由于continue的存在，会跳过寻找下一个起始点，直接结束循环，导致有连通分量被漏掉
        # if c > width or r > height or r < 0 or c < 0:
        #     continue
        if mask[r][c] == 0:
            #print(r,c,':',componentIndex)
            mask[r][c] = componentIndex
            filled.add((r, c))
            leftUp = (r - 1, c - 1)
            left = (r, c - 1)
            leftDown = (r + 1, c - 1)
            up = (r - 1, c)
            down = (r + 1, c)
            rightUp = (r - 1, c + 1)
            right = (r, c + 1)
            rightDown = (r + 1, c + 1)
            if leftUp not in filled and isInside(leftUp, (0, height),
                                                 (0, width)):
                fill.add(leftUp)
            if left not in filled and isInside(left, (0, height), (0, width)):
                fill.add(left)
            if leftDown not in filled and isInside(leftDown, (0, height),
                                                   (0, width)):
                fill.add(leftDown)
            if up not in filled and isInside(up, (0, height), (0, width)):
                fill.add(up)
            if down not in filled and isInside(down, (0, height), (0, width)):
                fill.add(down)
            if rightUp not in filled and isInside(rightUp, (0, height),
                                                  (0, width)):
                fill.add(rightUp)
            if right not in filled and isInside(right, (0, height),
                                                (0, width)):
                fill.add(right)
            if rightDown not in filled and isInside(rightDown, (0, height),
                                                    (0, width)):
                fill.add(rightDown)
        # print(fill)
        # 若fill中此时没有别的点了，标明上下左右邻近范围内的点都已被搜索完，则已经完成一个连通分量的搜索
        # 需要进行下一个连通分量的搜索
        if not fill:
            nextPoint = getNextStartPoint(mask)
            #print('next:',nextPoint)
            if nextPoint:
                fill.add(nextPoint)
                componentIndex = componentIndex + 1
    return mask, componentIndex


def getLargestConnectedComponent(mask, componetIndex) -> np.ndarray:
    """获取最大连通分量"""
    largestComponent = np.zeros_like(mask)
    # *此处生成的componentNumlist序列
    # *由于range(1,componetIndex)不包含componetIndex
    # *会缺少mask中元素值等于compoentIndex的点的数量
    # *将range(1,componetIndex)改为range(1,componetIndex+1)
    componetNumlist = [
        np.argwhere(mask == i).shape[0] for i in range(1, componetIndex + 1)
    ]
    largestComponetIndex = np.argmax(componetNumlist) + 1
    largestComponent[mask == largestComponetIndex] = 1
    return largestComponent


def DDT(imgs: np.ndarray, net: torch.nn.Module) -> np.ndarray:
    """利用DDT算法，获取图片的最大连接分量

    Parameters
    ----------
    imgs : np.ndarray
        N张图片构成的数组，形状为NCHW
    net : torch.nn.Module
        使用的CNN网络
    modelParaPath : str
        模型参数的路径
    
    Returns
    ----------
    largestComps : 最大连接分量
    """
    activation = getActivation(net, imgs)
    prinCompVector = getPrinCompVector(activation)
    # largestComps = []
    largestComps = np.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3]),
                            dtype=np.int8)
    for i in range(imgs.shape[0]):
        print('Processing pic {}/{}'.format(i + 1, imgs.shape[0]))
        indicatorMat = getIndicatorMatrix(activation, i, prinCompVector)
        reSizedIndicator = reSize(indicatorMat)
        mask, componetIndex = getConnectedComponet(reSizedIndicator)
        largestComp = getLargestConnectedComponent(mask, componetIndex)
        # largestComps.append(largestComp)
        largestComps[i] = largestComp
    # largestComps = np.array(largestComps)
    return largestComps


def drawLargestComp(imgs: np.ndarray, largestComps: np.ndarray):
    """将图片与最大连接分量进行绘制

    Parameters
    ----------
    imgs : np.ndarray
        N张图片构成的数组，形状为NCHW
    largestComps : np.ndarray
        N张图片的最大连接分量
    """
    # !图片显示时，为0的像素点显示为黑色，为1的像素点显示为白色，
    # !所以如果直接绘图的话，提取到的最大连通向量会显示为白色，无关的背景会显示为黑色
    plt.figure(figsize=(3.6 * 2, 4 * imgs.shape[0]))
    for i in range(imgs.shape[0]):
        plt.subplot(imgs.shape[0], 2, 2 * i + 1)
        plt.imshow(imgs[i, 0], cmap='gray')
        # plt.suptitle(str(i))
        plt.xticks(())
        plt.yticks(())
        plt.subplot(imgs.shape[0], 2, 2 * i + 2)
        # plt.suptitle(str(i))
        # ?// 1-largestComps[i]使得黑白反转，这样最大连通分量就显示为黑色
        plt.imshow(largestComps[i], cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.show()


def getDDTIndicatorMat(imgs: np.ndarray, net: torch.nn.Module) -> np.ndarray:
    """截取DDT算法中的一部分结果，获取指示矩阵

    Parameters
    ----------
    imgs : np.ndarray
        N张图片构成的数组，形状为NCHW
    net : torch.nn.Module
        使用的CNN网络

    Returns
    -------
    np.ndarray
        得到的指示矩阵，形状为NHW
    """
    activation = getActivation(net, imgs)
    prinCompVector = getPrinCompVector(activation)
    #此处indicatorMats的类型之前为np.int8，导致结果出现错误
    indicatorMats = np.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3]))
    for i in range(imgs.shape[0]):
        print('Processing pic {}/{}'.format(i + 1, imgs.shape[0]))
        indicatorMat = getIndicatorMatrix(activation, i, prinCompVector)
        reSizedIndicator = reSize(indicatorMat)
        indicatorMats[i] = reSizedIndicator
    return indicatorMats


def getReverseLargeComp(largecomp: np.ndarray) -> np.ndarray:
    """对最大连接向量进行黑白反转，并重新提取最大连接向量

    Parameters
    ----------
    largecomp : np.ndarray
        原本的最大连接向量，形状为NHW

    Returns
    -------
    np.ndarray
        翻转并重新提取的最大连接向量，形状为NHW
    """

    reverseLargeComp = 1 - largecomp
    reverseLargeCompArray = np.zeros_like(reverseLargeComp)
    for i in range(reverseLargeComp.shape[0]):
        mask, componentIndex = getConnectedComponet(reverseLargeComp[i])
        reverlargecomp = getLargestConnectedComponent(mask, componentIndex)
        reverseLargeCompArray[i] = reverlargecomp
    return reverseLargeCompArray
