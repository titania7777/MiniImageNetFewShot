import torch
import torch.nn.functional as F
import torch.nn as nn

class Self_Attn(nn.Module):
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
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

class Models(nn.Module):
    def __init__(self, mode='scratch', train_linear=True, num_class=64, way=5, shot=1, query=15, attention=True):
        super().__init__()
        self.way = way
        self.shot = shot
        self.query = query

        self.mode = mode
        self.train_linear = train_linear
        self.use_attention = attention

        if self.mode == 'scratch':
            self.linear = nn.Linear(1600, num_class)

        if self.use_attention:
            self.attention = Self_Attn(5)
        
        if not self.train_linear:
            self.scaler = nn.Parameter(torch.tensor(10.))

        self.encoder = nn.Sequential(
            self.conv_block(3, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
        )

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
        shot, query = x[:self.way*self.shot], x[self.way*self.shot:]
        
        # make prototype
        shot = shot.view(self.shot, self.way, -1).mean()
        
        shot = F.normalize(shot, dim=-1)
        query = F.normalize(query, dim=-1)

        return torch.mm(query, shot.t())

    def forward(self, shot, query):
        x = torch.cat((shot, query), dim=0)
        x = self.encoder(x)

        if self.use_attention:
            x = self.attention(x)
        
        if self.mode == 'scratch':
            x = self.cosine(x)*self.scaler
        else:
            if self.train_linear:
                x = self.linear(x.view(x.size(0), -1))
            else:
                x = self.cosine(x)*self.scaler

        return x
