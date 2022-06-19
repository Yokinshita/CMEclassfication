import torch
import torch.nn as nn
import torchvision.transforms as transforms


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
        self.fc = nn.Sequential(nn.Linear(50 * 53 * 53, 120),
                                nn.ReLU(),
                                nn.Dropout(drop_prob),
                                nn.Linear(120, 84),
                                nn.ReLU(),
                                nn.Dropout(drop_prob),
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
