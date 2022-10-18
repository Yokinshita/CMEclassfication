from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata
import cv2
import utils
from typing import Union


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
        # print('Processing pic {}/{}'.format(i + 1, imgs.shape[0]))
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


def pca(features):
    '''对特征图进行主成分分析，得到经过第一主成分投影变换的特征图

    Parameters
    ----------
    features : torch.Tensor
        需要进行PCA的特征图，形状为NCHW

    Returns
    -------
    torch.Tensor
        经过PCA变换的特征图，形状为NHW
    '''
    k = features.shape[0] * features.shape[2] * features.shape[3]
    x_mean = (features.sum(dim=2).sum(dim=2).sum(dim=0) /
              k).unsqueeze(0).unsqueeze(2).unsqueeze(2)
    features = features - x_mean

    reshaped_features = features.contiguous().view(features.shape[0], features.shape[1], -1)\
        .permute(1, 0, 2).contiguous().view(features.shape[1], -1)

    cov = torch.matmul(reshaped_features, reshaped_features.t()) / k
    # torch.eig函数将在未来版本被弃用，修改为torch.linalg.eig
    # eigval, eigvec = torch.eig(cov, eigenvectors=True)
    # first_compo = eigvec[:, 0]

    eigval, eigvec = torch.linalg.eig(cov)
    first_compo = eigvec.real[:, 0]

    projected_map = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.shape[0], -1)\
        .view(features.shape[0], features.shape[2], features.shape[3])

    maxv = projected_map.max()
    minv = projected_map.min()

    projected_map *= (maxv + minv) / torch.abs(maxv + minv)

    return projected_map


def DDTThirdParty(imgs: np.ndarray, net: torch.nn.Module):
    '''对imgs进行DDT算法，返回经过投影变换后的特征图

    Parameters
    ----------
    imgs : np.ndarray
        需要进行变换的图像，形状为NCHW
    net : torch.nn.Module
        使用的CNN网络

    Returns
    -------
    np.ndarray
        经过投影变换后的特征图，形状为NCHW
    '''
    features = torch.from_numpy(getActivation(net, imgs))
    features = utils.NHWCtoNCHW(features)  # pca的输入应当为HCHW
    project_map = torch.clamp(pca(features), min=0)
    maxv = project_map.view(project_map.size(0),
                            -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    project_map /= maxv

    project_map = F.interpolate(
        project_map.unsqueeze(1),
        size=(imgs.shape[2], imgs.shape[3]),
        mode='nearest',
    ) * 255.

    return project_map.detach().numpy()


def colorMapImage(imgs: np.ndarray,
                  project_map: np.ndarray,
                  mode=cv2.COLORMAP_JET):
    '''以DDT算法得到的投影特征图为mask，遮罩到原图片上。

    Parameters
    ----------
    imgs : np.ndarray
        原图片，形状为NCHW
    project_map : np.ndarray
        DDT算法得到的投影特征图，形状为NHW
    
    Returns
    ----------
    output_imgs : np.ndarray
        原图片和DDT算法得到的投影特征图遮罩到一起生成的图片，形状为NHWC
    '''

    output_imgs = np.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3], 3),
                           dtype=np.uint8)
    # 为了防止错误，强制输入的imgs为np.uint8类型
    if not np.issubdtype(imgs.dtype, np.uint8):
        raise ValueError(
            'Input imgs dtype only accept np.uint8 , but got {}'.format(
                imgs.dtype))
    for i in range(imgs.shape[0]):
        # img = cv2.resize(cv2.imread(os.path.join('./data', name)), (224, 224)) #读取为BGR格式
        # 将project_map repeat为(3,H,W)再转置为(H,W,3)
        # 这里的mask类似于自己的IndicatorMat
        mask = np.tile(project_map[i], reps=(3, 1, 1)).transpose(1, 2, 0)
        # *cv2默认图像格式为BGR，cvtColor转换得到的格式也为BGR，
        # *因此交给plt显示时需要转换为RGB格式，否则会产生色差
        mask = cv2.cvtColor(cv2.applyColorMap(mask.astype(np.uint8), mode),
                            cv2.COLOR_BGR2RGB)
        # addWeighted接受的两个数组应当为同样形状：HWC，因此将imgs[i]转为HWC形状
        # *output_imgs比较暗的原因是img在[0,1]区间，与mask加权相乘后基本不影响最后的图片
        img = np.tile(imgs[i], (3, 1, 1)).transpose(1, 2, 0).astype('uint8')
        output_img = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)
        output_imgs[i] = output_img
    return output_imgs


def toColorMap(project_maps: np.ndarray, mode=cv2.COLORMAP_JET):
    '''将图像按照数值映射为颜色色阶图

    Parameters
    ----------
    project_map : np.ndarray
        需要被映射的数组，形状为NCHW

    Returns
    -------
    np.ndarray
        颜色图，形状为NHWC
    '''
    colored_maps = np.zeros((project_maps.shape[0], project_maps.shape[2],
                             project_maps.shape[3], 3),
                            dtype=np.uint8)
    for i in range(project_maps.shape[0]):
        mask = np.tile(project_maps[i], reps=(3, 1, 1)).transpose(1, 2, 0)
        # *cv2默认图像格式为BGR，cvtColor转换得到的格式也为BGR，因此交给plt显示时需要转换为RGB格式
        mask = cv2.cvtColor(cv2.applyColorMap(mask.astype(np.uint8), mode),
                            cv2.COLOR_BGR2RGB)
        colored_maps[i] = mask
    return colored_maps
