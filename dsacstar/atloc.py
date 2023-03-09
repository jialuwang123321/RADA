import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torchvision import transforms, models
from dsacstar.att import AttentionBlock

class FourDirectionalLSTM(nn.Module):
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()
        self.feat_size = origin_feat_size // seq_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.lstm_rightleft = nn.LSTM(self.feat_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_downup = nn.LSTM(self.seq_size, self.hidden_size, batch_first=True, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_size).to(device),
                torch.randn(2, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        batch_size = x.size(0)
        x_rightleft = x.view(batch_size, self.seq_size, self.feat_size)
        x_downup = x_rightleft.transpose(1, 2)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        _, (hidden_state_lr, _) = self.lstm_rightleft(x_rightleft, hidden_rightleft)
        _, (hidden_state_ud, _) = self.lstm_downup(x_downup, hidden_downup)
        hlr_fw = hidden_state_lr[0, :, :]
        hlr_bw = hidden_state_lr[1, :, :]
        hud_fw = hidden_state_ud[0, :, :]
        hud_bw = hidden_state_ud[1, :, :]
        return torch.cat([hlr_fw, hlr_bw, hud_fw, hud_bw], dim=1)

class AtLoc(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=2048, lstm=False):
        super(AtLoc, self).__init__()
        self.droprate = droprate
        self.lstm = lstm

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        if self.lstm:
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=feat_dim, hidden_size=256)
            self.fc_xyz = nn.Linear(feat_dim // 2, 3)
            self.fc_wpqr = nn.Linear(feat_dim // 2, 3)
        else:
            self.att = AttentionBlock(feat_dim)
            self.fc_xyz = nn.Linear(feat_dim, 3)
            self.fc_wpqr = nn.Linear(feat_dim, 3)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)

        if self.lstm:
            x = self.lstm4dir(x)
        else:
            x = self.att(x.view(x.size(0), -1))

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)

class AtLocPlus(nn.Module):
    def __init__(self, atlocplus):
        super(AtLocPlus, self).__init__()
        self.atlocplus = atlocplus

    def forward(self, x):
        s = x.size()
        x = x.view(-1, *s[2:])
        poses = self.atlocplus(x)
        poses = poses.view(s[0], s[1], -1)
        return poses
    


# atloc = AVLoc(feature_extractor=models.resnet34(pretrained=True), droprate=opt.train_dropout, pretrained=True, lstm=opt.lstm)   
class AVLoc(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=2048, lstm=False):
        super(AVLoc, self).__init__()
        self.droprate = droprate
        self.lstm = lstm

        #feature_extractor = models.resnet34(pretrained=True)

        # replace the last FC layer in feature extractor
        #IMG model
        self.feature_extractor1 = feature_extractor
        self.feature_extractor1.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor1.fc.in_features
        self.feature_extractor1.fc = nn.Linear(fe_out_planes, feat_dim)

        #AUD model
        self.feature_extractor2 = feature_extractor
        self.feature_extractor2.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor2.fc.in_features
        self.feature_extractor2.fc = nn.Linear(fe_out_planes, feat_dim)
        

        if self.lstm:
            self.lstm4dir1 = FourDirectionalLSTM(seq_size=32, origin_feat_size=feat_dim, hidden_size=256)
            # self.lstm4dir2 = FourDirectionalLSTM(seq_size=32, origin_feat_size=feat_dim, hidden_size=256)
            self.fc_xyz = nn.Linear(feat_dim // 2, 3)
            self.fc_wpqr = nn.Linear(feat_dim // 2, 3)
        else: #这里
            self.att1 = AttentionBlock(feat_dim) #IMG model
            self.att2 = AttentionBlock(feat_dim) #AUD model
            self.fc_xyz = nn.Linear(feat_dim, 3)
            self.fc_wpqr = nn.Linear(feat_dim, 3)

        # initialize
        if pretrained:
            init_modules1 = [self.feature_extractor1.fc, self.fc_xyz, self.fc_wpqr] #IMG model
            init_modules2 = [self.feature_extractor1.fc, self.fc_xyz, self.fc_wpqr] #AUD model
        else:
            init_modules1 = self.modules() #IMG model
            init_modules2 = self.modules() #AUD model

        for m in init_modules1: #IMG model
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight.data) #initialization weights with normal distribution
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        
        for m in init_modules2: #AUD model 
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight.data) #initialization weights with normal distribution
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x1, x2):
        #IMG model
        x1 = self.feature_extractor1(x1)
        x1 = F.relu(x1)

        if self.lstm:
            x1 = self.lstm4dir1(x1)
        else:
            x1 = self.att1(x1.view(x1.size(0), -1))

        if self.droprate > 0:
            x1 = F.dropout(x1, p=self.droprate)

        #AUD model
        x2 = self.feature_extractor1(x2)
        x2 = F.relu(x2)

        if self.lstm:
            x2 = self.lstm4dir1(x2)
        else:
            x2 = self.att1(x2.view(x2.size(0), -1))

        if self.droprate > 0:
            x2 = F.dropout(x2, p=self.droprate)

        #Feature Combination
        x = x1 + x2

        #Common output
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)
