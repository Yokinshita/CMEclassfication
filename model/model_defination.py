import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Union, List, Any, Dict


class Net(nn.Module):
    def __init__(self, num_classes=2, drop_prob=0.5):
        super().__init__()
        # input_size 1*224*224
        self.drop_prob = drop_prob
        self.resize = transforms.Resize(224)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),  # out 96*26*26
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # out:96*26*26
        )
        self.conv2 = nn.Sequential(
            # in 96*26*26
            nn.Conv2d(96, 256, 5, 1, 2),  # out 256*26*26
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # out 256*12*12
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),  # out 384*12*12
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),  # out 384*12*12
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),  # out 384*12*12
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # out 256*5*5
        )
        self.fc = nn.Sequential(
            # in 6400*1
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.resize(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.fc(x.view(x.shape[0], -1))
        return out

    def load_param(self, model_path):
        map_location = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_state_dict(torch.load(model_path, map_location))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            out = self.forward(x)
            out = torch.argmax(out, dim=1)
        return out


class LeNet5(nn.Module):
    def __init__(self, num_classes=2, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.resize = transforms.Resize(224)
        #input N*1*224*224
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, 5),  #out N*20*220*220
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  #out N*20*110*110
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 50, 5),  #N*50*106*106
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  #N*50*53*53
        self.fc = nn.Sequential(nn.Linear(50 * 53 * 53, 120), nn.ReLU(),
                                nn.Dropout(drop_prob), nn.Linear(120, 84),
                                nn.ReLU(), nn.Dropout(drop_prob),
                                nn.Linear(84, num_classes))

    def forward(self, x):
        x = self.resize(x)
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.fc(x.view(x.shape[0], -1))
        return out

    def load_param(self, model_path):
        map_location = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_state_dict(torch.load(model_path, map_location))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            out = self.forward(x)
            out = torch.argmax(out, dim=1)
        return out


class VGG(nn.Module):
    def __init__(self,
                 features,
                 num_classes=2,
                 drop_prob=0.5,
                 init_weights=True):
        # input 3*512*512
        super(VGG, self).__init__()
        self.drop_prob = drop_prob
        self.features = features  # out 512*32*32
        self.classifier = nn.Sequential(
            nn.Linear(512 * 32 * 32, 4096),
            nn.ReLU(True),
            nn.Dropout(self.drop_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(self.drop_prob),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def load_param(self, model_path):
        map_location = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_state_dict(torch.load(model_path, map_location))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            out = self.forward(x)
            out = torch.argmax(out, dim=1)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _make_layers(cfg: List[Union[str, int]],
                 batch_norm: bool = False) -> nn.Sequential:
    '''用以构造VGG网络中卷积层的函数

    Parameters
    ----------
    cfg : List[Union[str, int]]
        包含卷积层网络构成的List
    batch_norm : bool, optional
        决定是否启用batch_norm，若为True，则启用，若为False，则不启用

    Returns
    -------
    nn.Sequential
        VGG网络中的卷积层构成的Sequential，作为VGG的feature
    '''
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512
    ],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, **kwargs: Any) -> VGG:
    model = VGG(_make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(**kwargs: Any) -> VGG:
    return _vgg('vgg11', 'A', False, **kwargs)


def vgg11_bn(**kwargs: Any) -> VGG:
    return _vgg('vgg11_bn', 'A', True, **kwargs)


def vgg13(**kwargs: Any) -> VGG:
    return _vgg('vgg13', 'B', False, **kwargs)


def vgg13_bn(**kwargs: Any) -> VGG:
    return _vgg('vgg13_bn', 'B', True, **kwargs)


def vgg19(**kwargs: Any) -> VGG:
    return _vgg('vgg19', 'E', False, **kwargs)


def vgg19_bn(**kwargs: Any) -> VGG:
    return _vgg('vgg19', 'E', True, **kwargs)
