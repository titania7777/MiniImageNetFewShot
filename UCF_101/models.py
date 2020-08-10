import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class Model(nn.Module):
    def __init__(self, num_classes, hidden_size=1024, num_layers=1, bidirectional=True, attention=True):
        super(Model, self).__init__()
        # Encoder(freeze)
        resnet = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self._freeze(self.encoder)
        
        # Adaptive Linear
        self.adaptive_linear = nn.Sequential(
            nn.Linear(resnet.fc.in_features, resnet.fc.in_features),
            nn.BatchNorm1d(resnet.fc.in_features),
            nn.ReLU(),
        )
        self.adaptive_linear.apply(self._initialize)
        
        # LSTM
        self.lstm = nn.LSTM(resnet.fc.in_features, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden = None
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_size if bidirectional else hidden_size, hidden_size if bidirectional else int(hidden_size/2)),
            nn.BatchNorm1d(hidden_size if bidirectional else int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(hidden_size if bidirectional else int(hidden_size/2), num_classes),
        )
        self.classifier.apply(self._initialize)

        # linear attention
        self.use_attention = attention
        self.attention = nn.Linear(2 * hidden_size if bidirectional else hidden_size, 1)
        self.attention.apply(self._initialize)

    def init_hidden(self):
        self.hidden = None

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = self.encoder(x.view(b*s, c, h ,w))
        x = self.adaptive_linear(x.view(b*s, -1))
        # x = x.view(x.size(0), x.size(1), -1).mean(-1)
        x, self.hidden = self.lstm(x.view(b, s, -1), self.hidden)
        if self.use_attention:
            attention = F.softmax(self.attention(x).squeeze(-1), dim=-1)
            x = torch.sum(attention.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]# last hidden
        x = self.classifier(x)
        return x
    
    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    # only linear
    def _initialize(self, model):
        if type(model) == nn.Linear:
            nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.01)