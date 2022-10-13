import torch
from PIL import Image
import numpy as np
import os
import os.path
from torchvision.transforms import functional as F
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Sequence
from typing import Union



def read_imgarray_from_singlepic(path_to_pic: str, transform) -> torch.Tensor:
    '''读取图片，并返回torch.tensor数组，形状为NCHW
    返回的tensor数组范围已经转换为[0,1]，
    Parameters
    ----------
    path_to_pic : str
        图片的路径
    transform :
        对图片施加的变换

    Returns
    -------
    torch.Tensor
        图片数组，形状为NCHW，类型为torch.float32
    '''
    # png图片是P模式，要转换为L模式
    img = Image.open(path_to_pic).convert('L')
    img = F.to_tensor(img)
    img = img.unsqueeze(dim=0)
    if transform:
        img = transform(img)
    return img


def read_imgarray_from_folder(path_to_folder: str, transform):
    """
    从单个文件夹中读取所有的图片，并转换为torch.tensor形式返回
    Arguments:
    ---------
    path_to_folder   : 数据所属的根文件夹路径
    Returns:
    -------
    imgarray         : 某个文件夹下所有图片组成的数组，形状为数量*通道*高*宽，形状为torch.float32
    """

    pics = os.listdir(path_to_folder)
    imglist = []
    for file in pics:
        img = read_imgarray_from_singlepic(os.path.join(path_to_folder, file),
                                           transform)
        imglist.append(img)
    imgtensor = torch.cat(imglist)
    return imgtensor


def load_CME(save_location, selected_remarks, transform):
    """
    将CME中selected_labels文件夹中的图片全部读取为imgarray
    Arguments:
    ---------
    path_to_folder   : 数据所属的根文件夹路径
    selected_labels  : 需要的数据所属的标签
    Returns:
    -------                                   
    imgarrays         : 所有图片组成的数组，形状为数量*通道*高*宽，类型为torch.float32
    labels            : 数据的标签，1表示CME，0表示非CME，类型为torch.int64
    """
    CME_path = os.path.join(save_location, 'CME')
    imglist = []
    for remark in selected_remarks:
        current_label_imgarray = read_imgarray_from_folder(
            os.path.join(CME_path, remark), transform)
        print('Reading CME data from {}'.format(os.path.join(CME_path,
                                                             remark)))
        imglist.append(current_label_imgarray)
    imgarray = torch.cat(imglist)
    labels = torch.ones(imgarray.shape[0], dtype=torch.int64)
    return imgarray, labels


def load_no_CME(save_location, transform):
    """
    加载非CME图片数据
    Arguments:
    ---------
    path_to_folder   : 数据所属的根文件夹路径
    Returns:
    -------
    imgarrays         : 所有图片组成的数组，形状为数量*高*宽*通道，类型为torch.float32
    labels            : 数据的标签，1表示CME，0表示非CME
    """

    No_CME_path = os.path.join(save_location, 'No CME')
    print('Reading No CME data from {}'.format(No_CME_path))
    imgarray = read_imgarray_from_folder(No_CME_path, transform)
    labels = torch.zeros(imgarray.shape[0], dtype=torch.int64)
    return imgarray, labels


class CMEdata:
    # 该类用以载入数据集，同时会将CME和非CME数据混合后打乱，并可以以TensorDataset形式输出
    def __init__(self, save_location: str, selected_remarks: list,
                 train_percentage: float, transform):
        """

        Arguments:
        ---------
        save_location       : 数据的根目录
        selected_remarks    : 需要用做数据集的图片所属的标签
        train_percentage    : 训练集的占比
        trans               : 对图片所作的变换

        """

        self.save_location = save_location
        self.selected_remarks = selected_remarks
        self.train_percentage = train_percentage
        self.transfrom = transform
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.size = None
        self.train_size = None
        self.test_size = None

    def __random_split(self, data: np.ndarray, labels: np.ndarray,
                       train_percentage: float):
        size = data.shape[0]  # 数据集中数据的个数
        shuffled_index = torch.randperm(size).tolist()
        split = int(train_percentage * size)  # 获得训练集和测试集的分划点
        # 0到split为训练集 split到最后为测试集
        train_index, test_index = shuffled_index[:split], shuffled_index[
            split:]
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

    def load_data_from_pic(self, train_percentage, save_npz, transform):
        """
        加载图片数据，打乱，分割并作为数据集和测试集
        ！！！！使用该方法，每次得到的训练集和测试集不完全相同
        Arguments:
        ---------
        train_percentage : 训练集所占全部数据集的比重，使用该方法时，需要自行划分训练集与测试集
        """
        print('Loading data from {}'.format(self.save_location))
        CMEdata, CME_labels = load_CME(self.save_location,
                                       self.selected_remarks, transform)
        no_CME_data, no_CME_labels = load_no_CME(self.save_location, transform)
        data = torch.cat((CMEdata, no_CME_data), axis=0)
        labels = torch.cat((CME_labels, no_CME_labels), axis=0)
        self.size = data.shape[0]
        self.train_size = int(self.size * train_percentage)
        self.test_size = self.size - self.train_size
        self.train_data, self.train_label, self.test_data, self.test_label = self.__random_split(
            data, labels, train_percentage)
        if save_npz:
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

    def load_data(self, forcing_load_from_pic, save_npz=False):
        npz_file_path = os.path.join(self.save_location, 'npz', 'data.npz')
        # 存在npz文件并且不强制从图片载入，则从npz文件载入
        if os.path.exists(npz_file_path) and forcing_load_from_pic is False:
            print('Pic npz file exists in {},load data from npz file'.format(
                npz_file_path))
            self.load_data_from_npz()
        else:
            self.load_data_from_pic(self.train_percentage, save_npz,
                                    self.transfrom)

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

        return torch.utils.data.TensorDataset(feature, label)


# *以下为新实现的数据集
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = Image.open(path).convert('L')
    return img


def has_file_allowed_extension(
        filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(
        extensions if isinstance(extensions, str) else tuple(extensions))


def folder_as_dataset(
    directory: str,
    class_index: int,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    '''将文件夹内所有的扩展名符合要求的文件路径和类编号作为元组返回

    Parameters
    ----------
    directory : str
        文件夹路径
    class_index : int
        类编号
    extensions : Optional[Union[str, Tuple[str, ...]]], optional
        符合要求的扩展名, by default None
    is_valid_file : Optional[Callable[[str], bool]], optional
        用以判断是否是符合要求的文件的函数, by default None

    Returns
    -------
    List[Tuple[str, int]]
        包含文件路径和类编号的列表，其中的每一个元素都是一个元组，每一个元组包含文件路径和类编号

    Raises
    ------
    ValueError
        当参数directory不是路径时触发
    ValueError
        当extensions和is_valid_file均为None时触发
    '''
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        raise ValueError(
            'parameter "directory" must be a directory,got {}'.format(
                directory))

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(
                x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []

    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                item = path, class_index
                instances.append(item)

    return instances


class Subset(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: torch.utils.data.Dataset
    indices: Sequence[int]
    _repr_indent = 4

    def __init__(self, dataset: torch.utils.data.Dataset,
                 indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        subset_samples = [self.dataset.samples[i] for i in indices]
        self.cls_nums = self._calculate_clsnums(subset_samples)

    @staticmethod
    def _calculate_clsnums(samples: List[Tuple[str, int]]) -> Dict[int, int]:
        '''返回一个包含samples中每个类和该类数量的字典

        Parameters
        ----------
        samples : List[Tuple[str, int]]
            包含图片路径和所属类别的元组构成的列表

        Returns
        -------
        Dict[int:int]
            类别和该类别的样本数量所构成的字典
        '''
        cls_nums: dict[int:int] = {}
        for filepath, class_idx in samples:
            if class_idx not in cls_nums.keys():
                cls_nums[class_idx] = 1
            cls_nums[class_idx] += 1
        return cls_nums

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __repr__(self) -> str:
        head = "Subset of " + self.dataset.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if hasattr(self.dataset,
                   "transforms") and self.dataset.transforms is not None:
            body += [repr(self.dataset.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


class CMEdataset(VisionDataset):
    """A generic data loader.

    Args:
        root (string): Root directory path.
        selected_remarks (List[str]): CME pics with selected_remarks would be loaded
        train_percentage(float):  percentage of size of train dataset to all dataset
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(
        self,
        root: str,
        selected_remarks: List[str],
        train_percentage: float,
        loader: Callable[[str], Any] = pil_loader,
        transform: Optional[Callable] = None,
        extensions: Optional[Tuple[str, ...]] = 'png',
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)
        CMEclass_to_idx = {
            selected_remark: 1
            for selected_remark in selected_remarks
        }
        CMEdir = [
            os.path.join(self.root, 'CME', selected_remark)
            for selected_remark in selected_remarks
        ]
        NoCMEdir = os.path.join(self.root, 'No CME')
        NoCMEclass_to_idx = 0
        CMEsamples = self.make_dataset(CMEdir, CMEclass_to_idx, extensions,
                                       is_valid_file)
        NoCMEsamples = self.make_dataset(NoCMEdir, NoCMEclass_to_idx,
                                         extensions, is_valid_file)
        self.selected_remarks = selected_remarks
        self.train_percentage = train_percentage
        self.loader = loader
        self.extensions = extensions

        self.samples = CMEsamples + NoCMEsamples
        self.targets = [s[1] for s in self.samples]

    def split(self, is_train):
        train_index, test_index = self._random_split(self.samples,
                                                     self.train_percentage)
        if is_train:
            return Subset(self, train_index)
        else:
            return Subset(self, test_index)

    @staticmethod
    def make_dataset(
        directory: Union[str, List[str]],
        class_to_idx: Union[Dict[str, int], int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        if isinstance(directory, str):
            if not isinstance(class_to_idx, int):
                raise ValueError(
                    'When directory is str , class_to_idx should be int')
            return folder_as_dataset(directory,
                                     class_to_idx,
                                     extensions=extensions,
                                     is_valid_file=is_valid_file)
        elif isinstance(directory, List):
            filelst = []
            for subdir in directory:
                dir_basename = os.path.basename(subdir)
                class_index = class_to_idx[dir_basename]
                filelst.extend(
                    folder_as_dataset(subdir,
                                      class_index,
                                      extensions=extensions,
                                      is_valid_file=is_valid_file))
            return filelst

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _random_split(samples, train_percentage: float) -> Tuple[List, List]:
        permed_indices = torch.randperm(len(samples)).tolist()
        split = int(len(samples) * train_percentage)
        train_index = permed_indices[:split]
        test_index = permed_indices[split:]
        return train_index, test_index


if __name__ == '__main__':
    # 该文件被设置为可以独立运行
    save_location = r'/home/lin/testdataset'
    selected_remarks = ['Only C2', 'Partial Halo', 'Poor Event']
    train_percentage = 0.7
    batch_size = 100
    trans = CenterCrop('NCHW')
    cmedata = CMEdata(save_location, selected_remarks, train_percentage, trans)
    cmedata.load_data(True)
    train_dataset = cmedata.to_tensordataset()
    train_iter = torch.utils.data.DataLoader(train_dataset,
                                             batch_size,
                                             shuffle=True)
    for X, y in train_iter:
        print(X.shape)
        print(y.shape)
    print('yes')
    print('finished')
