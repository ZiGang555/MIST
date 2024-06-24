import torch.nn as nn
import torch
from torch.nn.functional import interpolate
from models.layers import ConvLayer, get_norm, get_activation, EncoderBlock, Bottleneck, GlobalMaxPooling3D,DecoderBlock, VAEDecoderBlock
from models.aspp import CustomASPPModule
conv_kwargs = {"norm": "instance",
               "activation": "leaky",
               "negative_slope": 0.01,
               "down_type": "maxpool",
               "up_type": "transconv"}

class EdgeAttention(nn.Module):
    def __init__(self):
        super(EdgeAttention, self).__init__()

    def forward(self, M_i, T=0.5):
        # 计算边界热度图
        T = torch.full_like(M_i, fill_value=T)
        EA_i = 1 - torch.abs(M_i - T) / torch.max(T, 1 - T)
        return EA_i

class EdgeFeatureEnhanceModule(nn.Module):
    def __init__(self, decoder, **kwargs):
        super(EdgeFeatureEnhanceModule, self).__init__()
        self.edge_attention = EdgeAttention()
        # self.transconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.decoder  = decoder
    def forward(self, skip, M_i, E_i):
        # 计算边界热度图
        EA_i = self.edge_attention(M_i)
        # 对边缘特征图使用转置卷积上采样
        edge_feature = self.decoder(skip, E_i)
        # 特征图F_i与上采样后的边缘特征图相加
        enhanced_edge_feature = edge_feature + EA_i
        return enhanced_edge_feature
    

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ResNetBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, use_norm=True, use_activation=False, **kwargs)

        self.residual_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.residual_norm = get_norm(kwargs["norm"], out_channels, **kwargs)
        self.final_act = get_activation(kwargs["activation"], in_channels=out_channels, **kwargs)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.residual_norm(res)

        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.add(x, res)
        x = self.final_act(x)
        return x

class DualOutModel(nn.Module):

    def __init__(self,
                 block,
                 n_classes,
                 n_channels,
                 init_filters,
                 depth,
                 pocket,
                 deep_supervision,
                 deep_supervision_heads,
                 vae_reg,
                 latent_dim,
                 **kwargs):
        super(DualOutModel, self).__init__()

        # User defined inputs
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.init_filters = init_filters
        kwargs["groups"] = self.init_filters

        self.depth = depth
        self.pocket = pocket
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = deep_supervision_heads
        self.vae_reg = vae_reg
        self.latent_dim = latent_dim

        # Make sure number of deep supervision heads is less than network depth
        assert self.deep_supervision_heads < self.depth

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        # Encoder branch
        self.encoder = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                in_channels = self.n_channels
            else:
                in_channels = self.init_filters * self.mul_on_downsample ** (i - 1)

            out_channels = self.init_filters * self.mul_on_downsample ** i
            self.encoder.append(EncoderBlock(in_channels, out_channels, block, **kwargs))

        in_channels = self.init_filters * self.mul_on_downsample ** (self.depth - 1)
        out_channels = self.init_filters * self.mul_on_downsample ** self.depth
        self.bottleneck = Bottleneck(in_channels, out_channels, block, **kwargs)

        # VAE Regularization
        if self.vae_reg:
            self.normal_dist = torch.distributions.Normal(0, 1)
            self.normal_dist.loc = self.normal_dist.loc.cuda()
            self.normal_dist.scale = self.normal_dist.scale.cuda()

            self.global_maxpool = GlobalMaxPooling3D()
            self.mu = nn.Linear(self.init_filters * self.mul_on_downsample ** self.depth, self.latent_dim)
            self.sigma = nn.Linear(self.init_filters * self.mul_on_downsample ** self.depth, self.latent_dim)

            self.vae_decoder = nn.ModuleList()
            for i in range(self.depth - 1, -1, -1):
                if i == self.depth - 1:
                    in_channels = 1
                else:
                    in_channels = self.init_filters * self.mul_on_downsample ** (i + 1)
                out_channels = self.init_filters * self.mul_on_downsample ** i
                self.vae_decoder.append(VAEDecoderBlock(in_channels, out_channels, block, **kwargs))

            self.vae_out = nn.Conv3d(in_channels=self.init_filters, out_channels=self.n_channels, kernel_size=1)

        # Define main decoder branch
        self.decoder = nn.ModuleList()
        
        if self.deep_supervision:
            self.head_ids =[head for head in range(1, self.deep_supervision_heads + 1)]
            self.deep_supervision_out = nn.ModuleList()
        for i in range(self.depth - 1, -1, -1):
            in_channels = self.init_filters * self.mul_on_downsample ** (i + 1)
            out_channels = self.init_filters * self.mul_on_downsample ** i
            self.decoder.append(DecoderBlock(in_channels, out_channels, block, **kwargs))

            # Define pointwise convolutions for deep supervision heads
            if self.deep_supervision and i in self.head_ids:
                head = nn.Conv3d(in_channels=out_channels,
                                 out_channels=self.n_classes,
                                 kernel_size=1)
                self.deep_supervision_out.append(head)

        # self.decoder2 = nn.ModuleList()
        self.EA = nn.ModuleList()
        # if self.deep_supervision:
        #     self.head_ids =[head for head in range(1, self.deep_supervision_heads + 1)]
        #     self.deep_supervision_out = nn.ModuleList()
        for i in range(self.depth - 1, -1, -1):
            in_channels = self.init_filters * self.mul_on_downsample ** (i + 1)
            out_channels = self.init_filters * self.mul_on_downsample ** i
            # self.decoder2.append(DecoderBlock(in_channels, out_channels, block, **kwargs))
            self.EA.append(EdgeFeatureEnhanceModule(DecoderBlock(in_channels, out_channels, block, **kwargs)))



        # Define pointwise convolution for final output
        self.out1 = nn.Conv3d(in_channels=self.init_filters,
                             out_channels=self.n_classes,
                             kernel_size=1)
        self.out2 = nn.Conv3d(in_channels=self.init_filters,
                             out_channels=self.n_classes,
                             kernel_size=1)
        self.aspp = CustomASPPModule(in_dim=self.n_classes, edge_dim=self.n_classes, output_dim=self.n_classes)

    def forward(self, x):
        # Get current input shape for deep supervision
        input_shape = (int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))

        # Encoder
        skips = list()
        for encoder_block in self.encoder:
            skip, x = encoder_block(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)
        x2 = x

        # VAE Regularization
        if self.vae_reg and self.training:
            x_vae = self.global_maxpool(x)
            mu = self.mu(x_vae)
            log_var = self.sigma(x_vae)

            # Sample from distribution
            x_vae = mu + torch.exp(0.5 * log_var)*self.normal_dist.sample(mu.shape)

            # Reshape for decoder
            x_vae = torch.reshape(x_vae, (x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]))

            # Start VAE decoder
            for i, decoder_block in enumerate(self.vae_decoder):
                x_vae = decoder_block(x_vae)

            x_vae = self.vae_out(x_vae)
            output_vae = (x_vae, mu, log_var)

        # Add deep supervision heads
        if self.deep_supervision and self.training:
            deep_supervision_heads = list()

        # Decoder
        skips.reverse()
        for i, (skip, decoder_block, EA_block) in enumerate(zip(skips, self.decoder,self.EA)):
            x = decoder_block(skip, x)
            # 边缘分支
            x2 = EA_block(skip,x, x2)
            # x2 = decoder_block2(skip, x2)

            if self.deep_supervision and self.training and i in self.head_ids:
                deep_supervision_heads.append(x)

        # Apply deep supervision
        if self.deep_supervision and self.training:
            # Create output list
            output_deep_supervision = list()

            for head, head_out in zip(deep_supervision_heads, self.deep_supervision_out):
                current_shape = (int(head.shape[2]), int(head.shape[3]), int(head.shape[4]))
                scale_factor = tuple([int(input_shape[i] // current_shape[i]) for i in range(3)])
                head = interpolate(head, scale_factor=scale_factor, mode="trilinear")
                output_deep_supervision.append(head_out(head))

            output_deep_supervision.reverse()
            output_deep_supervision = tuple(output_deep_supervision)


        if self.training:
            output = dict()
            out1 = self.out1(x)
            out2 = self.out2(x2)
            out1 = self.aspp(out1, out2)
            output["prediction"] = out1
            output["edge"] = out2

            if self.deep_supervision:
                output["deep_supervision"] = output_deep_supervision

            if self.vae_reg:
                output["vae_reg"] = output_vae

        else:
            out1 = self.out1(x)
            out2 = self.out2(x2)
            output = self.aspp(out1, out2)
            # output = self.out(x)

        return output




        return output

class DualOutNet(nn.Module):

    def __init__(self,
                 n_classes,
                 n_channels,
                 init_filters,
                 depth,
                 pocket,
                 deep_supervision,
                 deep_supervision_heads,
                 vae_reg,
                 latent_dim):
        super(DualOutNet, self).__init__()

        self.base_model = DualOutModel(ResNetBlock,
                                    n_classes,
                                    n_channels,
                                    init_filters,
                                    depth,
                                    pocket,
                                    deep_supervision,
                                    deep_supervision_heads,
                                    vae_reg,
                                    latent_dim,
                                    **conv_kwargs)

    def forward(self, x, **kwargs):
        return self.base_model(x, **kwargs)
