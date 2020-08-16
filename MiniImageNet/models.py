import torch
import torch.nn.functional as F
import torch.nn as nn

class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        m_batchsize ,C ,width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0, 2, 1) )
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma*out + x
        return out

class Model(nn.Module):
    def __init__(self, mode, num_classes=64, way=5, shot=1, query=15, attention=True):
        super(Model, self).__init__()
        self.way = way
        self.shot = shot
        self.query = query

        self.mode = mode
        self.use_attention = attention

        if not self.mode:
            self.linear = nn.Linear(1600, num_classes)

        if self.use_attention:
            self.attention = Self_Attn(64)

        # conv4-64F, you can change to anything you want
        self.encoder = nn.Sequential(
            self.conv_block(3, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
        )

        self.shot_scaler = nn.Parameter(torch.tensor(2.))
        self.query_scaler = nn.Parameter(torch.tensor(2.))

    def conv_block(self, in_channels, out_channels):
        bn = nn.BatchNorm2d(out_channels)
        nn.init.uniform_(bn.weight)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            bn,
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )

    def cosine(self, x):
            batch_size = x.size(0)
            shot, query = x[:, :self.way*self.shot], x[:, self.way*self.shot:]

            # make prototype
            shot = shot.view(batch_size, self.shot, self.way, -1).mean(dim=1)

            shot = F.normalize(shot, dim=-1)*self.shot_scaler
            query = F.normalize(query, dim=-1)*self.query_scaler

            return torch.bmm(query, shot.transpose(1, 2)).reshape(-1, self.way)

    def forward(self, x, linear):
        batch_size = x.size(0)
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.encoder(x)

        if self.use_attention:
            x = self.attention(x)

        if not self.mode and linear:
            x = self.linear(x.view(x.size(0), -1))
        else:
            x = x.view(batch_size, self.way*self.query + self.way*self.shot, -1)
            x = self.cosine(x)

        return x
