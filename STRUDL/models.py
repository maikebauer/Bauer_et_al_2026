
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calc_conv(w,k,s,p):

    sz = ((w-k+2*p)/s).astype(int)+1

    return tuple(sz)

def calc_kernel(inp_size, kernel_size, depth):

    kernels = []
    outputs = np.zeros((depth+1, 3))

    outputs[0] = inp_size

    stride = 2
    padding = 0

    for i in range(1,depth+1):

        ks = outputs[i-1] % kernel_size
        ks[ks == 0] = kernel_size
        kernels.append(tuple(map(int, ks)))

        outputs[i] = calc_conv(outputs[i-1], kernels[i-1], s=stride, p=padding)

    return kernels

class CNN3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN3D, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        # Encoder layers

        self.encoder_conv_00 = nn.Sequential(*[nn.Conv3d(in_channels=self.input_channels,out_channels=64,kernel_size=7,padding=3)])
        self.encoder_conv_01 = nn.Sequential(*[nn.Conv3d(in_channels=64,out_channels=64,kernel_size=7,padding=3)])

        self.encoder_conv_10 = nn.Sequential(*[nn.Conv3d(in_channels=64,out_channels=128,kernel_size=7,padding=3)])
        self.encoder_conv_11 = nn.Sequential(*[nn.Conv3d(in_channels=128,out_channels=128,kernel_size=7,padding=3)])

        self.encoder_conv_20 = nn.Sequential(*[nn.Conv3d(in_channels=128,out_channels=256,kernel_size=3,padding=1)])
        self.encoder_conv_21 = nn.Sequential(*[nn.Conv3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.encoder_conv_22 = nn.Sequential(*[nn.Conv3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])

        self.encoder_conv_30 = nn.Sequential(*[nn.Conv3d(in_channels=256,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_31 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_32 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        self.encoder_conv_40 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_41 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_42 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])


        # Decoder layers

        self.decoder_convtr_42 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_41 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_40 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        self.decoder_convtr_32 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_31 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_30 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=256,kernel_size=3,padding=1)])

        self.decoder_convtr_22 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.decoder_convtr_21 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.decoder_convtr_20 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=256,out_channels=128,kernel_size=3,padding=1)])

        self.decoder_convtr_11 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=128,out_channels=128,kernel_size=3,padding=1)])
        self.decoder_convtr_10 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=128,out_channels=64,kernel_size=3,padding=1)])

        self.decoder_convtr_01 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)])
        self.decoder_convtr_00 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=64,out_channels=self.output_channels,kernel_size=3,padding=1)])

        self.dropout = nn.Dropout3d()

        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU()

    def forward(self, input_img):

        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = F.relu(self.encoder_conv_00(input_img))
        x_01 = F.relu(self.encoder_conv_01(x_00))

        kernels = calc_kernel(x_01.size()[2:], kernel_size=2, depth=5)
        x_0, indices_0 = F.max_pool3d(x_01, kernel_size=kernels[0], stride=2, return_indices=True)

        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_conv_10(x_0))
        x_11 = F.relu(self.encoder_conv_11(x_10))
        x_1, indices_1 = F.max_pool3d(x_11, kernel_size=kernels[1], stride=2, return_indices=True)

        x_1 = self.dropout(x_1)

        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = F.relu(self.encoder_conv_20(x_1))
        x_21 = F.relu(self.encoder_conv_21(x_20))
        x_22 = F.relu(self.encoder_conv_22(x_21))
        x_2, indices_2 = F.max_pool3d(x_22, kernel_size=kernels[2], stride=2, return_indices=True)

        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = F.relu(self.encoder_conv_30(x_2))
        x_31 = F.relu(self.encoder_conv_31(x_30))
        x_32 = F.relu(self.encoder_conv_32(x_31))
        x_3, indices_3 = F.max_pool3d(x_32, kernel_size=kernels[3], stride=2, return_indices=True)

        x_3 = self.dropout(x_3)
        
        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = F.relu(self.encoder_conv_40(x_3))
        x_41 = F.relu(self.encoder_conv_41(x_40))
        x_42 = F.relu(self.encoder_conv_42(x_41))
        x_4, indices_4 = F.max_pool3d(x_42, kernel_size=kernels[4], stride=2, return_indices=True)

        # Decoder Stage - 5
        x_4d = F.max_unpool3d(x_4, indices_4, kernel_size=kernels[4], stride=2, output_size=dim_4)
        x_42d = F.relu(self.decoder_convtr_42(x_4d))
        x_41d = F.relu(self.decoder_convtr_41(x_42d))
        x_40d = F.relu(self.decoder_convtr_40(x_41d))

        x_40d = x_40d + x_3

        # Decoder Stage - 4
        x_3d = F.max_unpool3d(x_40d, indices_3, kernel_size=kernels[3], stride=2, output_size=dim_3)
        x_32d = F.relu(self.decoder_convtr_32(x_3d))
        x_31d = F.relu(self.decoder_convtr_31(x_32d))
        x_30d = self.dropout(F.relu(self.decoder_convtr_30(x_31d)))

        x_30d = x_30d + x_2

        # Decoder Stage - 3
        x_2d = F.max_unpool3d(x_30d, indices_2, kernel_size=kernels[2], stride=2, output_size=dim_2)
        x_22d = F.relu(self.decoder_convtr_22(x_2d))
        x_21d = F.relu(self.decoder_convtr_21(x_22d))
        x_20d = F.relu(self.decoder_convtr_20(x_21d))

        x_20d = x_20d + x_1

        # Decoder Stage - 2
        x_1d = F.max_unpool3d(x_20d, indices_1, kernel_size=kernels[1], stride=2, output_size=dim_1)
        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        x_10d = self.dropout(F.relu(self.decoder_convtr_10(x_11d)))

        x_10d = x_10d + x_0

        # Decoder Stage - 1
        x_0d = F.max_unpool3d(x_10d, indices_0, kernel_size=kernels[0], stride=2, output_size=dim_0)
        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d) 

        return self.sigmoid(x_00d)