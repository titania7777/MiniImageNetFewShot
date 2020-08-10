import torch.nn as nn
from torchvision.models import resnet18

class Model(nn.Module):
    def __init__(self, num_classes, hidden_size=1024, num_layers=1, bidirectional=True):
        super(Model, self).__init__()
        # Encoder(freeze)
        resnet = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self._freeze(self.encoder)
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
    
    def init_hidden(self):
        self.hidden = None

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = self.encoder(x.view(b*s, c, h ,w))
        # x = x.view(x.size(0), x.size(1), -1).mean(-1)
        x, self.hidden = self.lstm(x.view(b, s, -1), self.hidden)
        x = self.classifier(x[:, -1])# last hidden
        return x
    
    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    # only linear
    def _initialize(self, model):
        if type(model) == nn.Linear:
            nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.01)