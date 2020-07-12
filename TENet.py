import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,A:torch.Tensor):
        super(GCNLayer, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1= torch.nn.Parameter(torch.tensor([0.1],requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 =nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 =nn.Sequential( nn.Linear(input_dim, output_dim))
        nodes_count=self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask=torch.ceil( self.A*0.00001)
        
        
    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat
    
    def forward(self, H, model='normal'):
        H = self.BN(H)
        H_xx1= self.GCN_liner_theta_1(H)
        A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        if model != 'normal': A=torch.clamp(A,0.1) #This is a trick.
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)
        return output,A

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out

class TENet(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, model='normal'):
        super(TENet, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.CNN_denoise = nn.Sequential(
            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )
        
        self.CNN_Branch = nn.Sequential(
            SSConv(128, 128,kernel_size=5),
            SSConv(128, 64,kernel_size=5),
        )
        
        # GCN layers
        self.GCN_Layer_1 = GCNLayer(128, 128, self.A)
        self.GCN_Layer_2 = GCNLayer(128, 64, self.A)

        self.Softmax_linear =nn.Sequential(nn.Linear(128, self.class_count))
    
    def forward(self, x: torch.Tensor):
        '''
        :param x: C*H*W
        :return: probability_map H*W*C
        '''
        (h, w, c) = x.shape
        
        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise =torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x=noise 
        
        clean_x_flatten=clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten) 
        hx = clean_x
        
        # CNN与GCN分两条支路
        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))# spectral-spatial convolution
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q
        H = superpixels_flatten
        if self.model=='normal':
            H,A_1 = self.GCN_Layer_1(H)
            H,A_2 = self.GCN_Layer_2(H)
        else:
            H,A_1 = self.GCN_Layer_1(H,model='smoothed')
            H,A_2 = self.GCN_Layer_2(H,model='smoothed')
            
        GCN_result = torch.matmul(self.Q, H)
        
        # 两组特征融合(两种融合方式)
        Y = torch.cat([GCN_result,CNN_result],dim=-1)
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y,A_1,A_2

