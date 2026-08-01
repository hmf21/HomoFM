import math
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from utils.kde import kde
from utils.local_correlation import local_correlation
from utils.utils import get_autocast_params, get_tuple_transform_ops

from model.crossview_decoder_light import CrossVITDecoder_noself
from model.FPN import FPNEncoder, FPNDecoder_concat, Swish

from model.STFA import ChannelAttentionAggregation, CorrectedChannelAttentionAggregation, SpatialCrossAttentionFusion

class GFNet(nn.Module):
    def __init__(self,
                 conf,
                 sample_mode = "threshold_balanced",
                 exact_softmax = False,
                 amp=True,
                 amp_dtype=torch.float16,
                 initial_res=(448, 448),
                 upsample_res=(560, 560),
                 symmetric = False,
                 upsample_preds = False,
                 attenuate_cert=False,
                 use_small=True,
                 use_fm_head=True
                 ):
        super().__init__()
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.sample_mode = sample_mode
        self.exact_softmax = exact_softmax
        self.h_resized, self.w_resized = initial_res
        self.initial_res = initial_res
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res
        self.symmetric = symmetric
        self.attenuate_cert = attenuate_cert
        self.sample_thresh = 0.05

        args = conf
        self.num_grid = args['matcher']['num_grid']
        ## dino part
        vit_kwargs = dict(img_size=518,
                          patch_size=14,
                          init_values=1.0,
                          ffn_layer="mlp",
                          block_chunks=0,
                          )
        self.use_small = use_small
        if self.use_small:
            from .transformer import vit_small
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth", map_location="cpu")
            dinov2_vitl14 = vit_small(**vit_kwargs).eval()
        else:
            from .transformer import vit_large
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
            dinov2_vitl14 = vit_large(**vit_kwargs).eval()

        dinov2_vitl14.load_state_dict(dinov2_weights)
        for param in dinov2_vitl14.parameters():
            param.requires_grad = False
        self.dino = [dinov2_vitl14]
        self.dino_decoder = CrossVITDecoder_noself(conf=args, upsample=False)

        feature_dim = args['encoder_cfg']['feat_chs'] ## coarse to fine
        self.encoder = FPNEncoder(feat_chs=feature_dim[::-1])
        self.decoder = FPNDecoder_concat(feat_chs=feature_dim[::-1])

        final_dim = feature_dim[0]
        # self.merge_layer = nn.Sequential(nn.Conv2d(2*final_dim, final_dim, kernel_size=3, padding=1), nn.BatchNorm2d(final_dim), Swish())
        self.caa_merge_layer = SpatialCrossAttentionFusion(in_channels_s=final_dim, in_channels_t=final_dim)

        radius = args["matcher"]["radius"] ## coarse to fine
        self.radius = radius
        self.num_itr = args["matcher"]["num_itr"] ## coarse to fine
        displacement_dim = args["matcher"]["displacement_dim"] ## coarse to fine
        dw = True
        hidden_blocks = 8
        kernel_size = 5
        displacement_emb = "linear"
        disable_local_corr_grad = True
        use_amp = self.amp
        self.conv_refiner = nn.ModuleDict(
            {
                "16": ConvRefiner(
                    2*feature_dim[0] + displacement_dim[0] + (2*radius[0]+1)**2,
                    2*feature_dim[0] + displacement_dim[0] + (2*radius[0]+1)**2,
                    2 + 1,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=displacement_dim[0],
                    local_corr_num = radius[0],
                    corr_in_other = True,
                    amp = use_amp,
                    disable_local_corr_grad = disable_local_corr_grad,
                    bn_momentum = 0.01,
                    use_fm_head = use_fm_head,
                ),
                "8": ConvRefiner(
                    2*feature_dim[0] + displacement_dim[1] + (2*radius[1]+1)**2,
                    2*feature_dim[0] + displacement_dim[1] + (2*radius[1]+1)**2,
                    2 + 1,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=displacement_dim[1],
                    local_corr_num = radius[1],
                    corr_in_other = True,
                    amp = use_amp,
                    disable_local_corr_grad = disable_local_corr_grad,
                    bn_momentum = 0.01,
                    use_fm_head=use_fm_head,
                ),
                "4": ConvRefiner(
                    2*feature_dim[1] + displacement_dim[2] + (2*radius[2]+1)**2,
                    2*feature_dim[1] + displacement_dim[2] + (2*radius[2]+1)**2,
                    2 + 1,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=displacement_dim[2],
                    local_corr_num = radius[2],
                    corr_in_other = True,
                    amp = use_amp,
                    disable_local_corr_grad = disable_local_corr_grad,
                    bn_momentum = 0.01,
                    use_fm_head=use_fm_head,
                ),
                "2": ConvRefiner(
                    2*feature_dim[2] + displacement_dim[3] + (2*radius[3]+1)**2,
                    2*feature_dim[2] + displacement_dim[3] + (2*radius[3]+1)**2,
                    2 + 1,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=displacement_dim[3],
                    local_corr_num = radius[3],
                    corr_in_other = True,
                    amp = use_amp,
                    disable_local_corr_grad = disable_local_corr_grad,
                    bn_momentum = 0.01,
                    use_fm_head=use_fm_head,
                ),
                "1": ConvRefiner(
                    2*feature_dim[3] + displacement_dim[4],
                    2*feature_dim[3] + displacement_dim[4],
                    2 + 1,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks = hidden_blocks,
                    displacement_emb = displacement_emb,
                    displacement_emb_dim = displacement_dim[4],
                    local_corr_num = radius[4],
                    corr_in_other = False,
                    amp = use_amp,
                    disable_local_corr_grad = disable_local_corr_grad,
                    bn_momentum = 0.01,
                    use_fm_head=use_fm_head,
                ),
            }
        )
    def extract_features(self, x, upsample=False):
        twoB, C, H, W = x.shape
        vit_h, vit_w = int(H // 14 * 14), int(W // 14 * 14)

        if H != vit_h:
            vit_imgs = F.interpolate(x, (vit_h, vit_w), mode='bilinear',
                                     align_corners=False)
        else:
            vit_imgs = x
        with torch.no_grad():
            if self.dino[0].device != vit_imgs.device:
                self.dino[0] = self.dino[0].to(vit_imgs.device).to(self.amp_dtype)
            dinov2_features_14 = self.dino[0].forward_features(vit_imgs.to(self.amp_dtype))
            features_14 = dinov2_features_14['x_norm_patchtokens']
            del dinov2_features_14
        with torch.autocast(device_type="cuda", enabled=self.amp, dtype=self.amp_dtype):
            vit0, vit1 = self.dino_decoder(features_14.chunk(2)[0], features_14.chunk(2)[1], vit_shape=(twoB//2, C, vit_h//14, vit_w//14))
        vit_feat = torch.cat((vit0.float(), vit1.float()), dim=0)

        # 上采样对齐dino feat 和 CNN feat
        conv31_h, conv31_w = H // 8, W // 8
        if vit_feat.shape[2] != conv31_h or vit_feat.shape[3] != conv31_w:
            vit_feat_up = F.interpolate(vit_feat, size=(conv31_h, conv31_w), mode='bilinear',
                                        align_corners=False)
        else:
            vit_feat_up = vit_feat
        conv01, conv11, conv21, conv31 = self.encoder(x)
        # conv31 = conv31 + self.merge_layer(torch.cat((conv31, vit_feat_up), dim=1)) # 这里的融合方式是不是有点太简单了
        conv31 = self.caa_merge_layer(vit_feat_up, conv31)

        feat1, feat2, feat3, feat4 = self.decoder(conv01, conv11, conv21, conv31)

        f_q_pyramid = {
            "16": vit_feat.chunk(2)[0],
            "8": feat1.chunk(2)[0],
            "4": feat2.chunk(2)[0],
            "2": feat3.chunk(2)[0],
            "1": feat4.chunk(2)[0],
        }
        f_s_pyramid = {
            "16": vit_feat.chunk(2)[1],
            "8": feat1.chunk(2)[1],
            "4": feat2.chunk(2)[1],
            "2": feat3.chunk(2)[1],
            "1": feat4.chunk(2)[1],
        }
        if upsample:
            del f_q_pyramid["16"], f_s_pyramid["16"]
        return f_q_pyramid, f_s_pyramid

    def forward(self, batch, symmetric=False, upsample=False, scale_factor=1, pre_corresps=None, visualization=False):
        im0 = batch["im_A"]
        im1 = batch["im_B"]
        corresps = {}
        B, C, H0, W0 = im0.shape
        B, C, H1, W1 = im1.shape

        x = torch.cat([im0, im1], dim=0)
        features0, features1 = self.extract_features(x, upsample)
        all_scales = list(features0.keys())
        if symmetric:
            f_q_pyramid = {
                scale: torch.cat((features0[scale], features1[scale]), dim = 0)
                for scale in features0.keys()
            }
            f_s_pyramid = {
                scale: torch.cat((features1[scale], features0[scale]), dim = 0)
                for scale in features0.keys()
            }
            features0, features1 = f_q_pyramid, f_s_pyramid
        if upsample:
            num_grid = self.num_grid_up
            num_itr = self.num_itr_up
        else:
            num_grid = self.num_grid
            num_itr = self.num_itr

        for idx, scale in enumerate(features0.keys()):
            f0 = features0[scale]
            f1 = features1[scale]

            if scale == all_scales[0]:
                if upsample:
                    assert pre_corresps is not None, "you should provide a pre_corresps for upsampling refine."
                    flow, certainty = pre_corresps["flow"], pre_corresps["certainty"]
                    flow = F.interpolate(
                            flow,
                            size=num_grid[0],
                            align_corners=False,
                            mode="bilinear",
                        )
                    certainty = F.interpolate(
                            certainty,
                            size=num_grid[0],
                            align_corners=False,
                            mode="bilinear",
                        )
                else:
                    corr_volume = self.corr_volume(f0, f1)
                    flow = self.pos_embed(corr_volume) ## B 2 H W
                    certainty = torch.zeros_like(flow)[:, 0][:, None] ## B 1 H W

            corresps[scale] = {}
            displacement_pre = torch.zeros_like(flow) + 1e-7
            for itr in range(num_itr[idx]):

                delta_flow, delta_certainty, local_corr = self.conv_refiner[scale](
                    num_grid[idx], f0, f1, flow, scale_factor=scale_factor, logits=None,
                )
                displacement = int(scale) * torch.stack((delta_flow[:, 0].float() / (4 * W0),
                                                delta_flow[:, 1].float() / (4 * H0),),dim=1,)
                if not self.training:
                    displacement[((displacement-displacement_pre).abs()/(displacement_pre).abs())<1e-6] = 0
                flow = flow + displacement
                certainty = certainty + delta_certainty
                corresps[scale][itr+1] = {'flow': flow, 'certainty': certainty}
                displacement_pre = displacement

            if scale != '1':
                flow = F.interpolate(
                    flow,
                    size=num_grid[idx+1],
                    mode='bilinear',
                ).detach()
                certainty = F.interpolate(
                    certainty,
                    size=num_grid[idx+1],
                    mode='bilinear',
                ).detach()

        return corresps

    @torch.inference_mode()
    def match(self, im0, im1, *args, batched = True):
        if isinstance(im0, (str, Path)):
            im0, im1 = Image.open(im0).convert("RGB"), Image.open(im1).convert("RGB")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            im0 = ToTensor()(im0)[None].to(device)
            im1 = ToTensor()(im1)[None].to(device)
            imA, imB = im0, im1
            test_transform = get_tuple_transform_ops(
                resize=(self.h_resized, self.w_resized), mode=2, normalize=True, clahe = False
            )
            im0, im1 = test_transform((im0.squeeze(0), im1.squeeze(0)))
            im0 = im0.to(device).unsqueeze(0)
            im1 = im1.to(device).unsqueeze(0)
        elif isinstance(im0, Image.Image):
            batched = False
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            im0 = ToTensor()(im0)[None].to(device)
            im1 = ToTensor()(im1)[None].to(device)
            imA, imB = im0, im1
            test_transform = get_tuple_transform_ops(
                resize=(self.h_resized, self.w_resized), normalize=True, clahe = False
            )
            im0, im1 = test_transform((im0.squeeze(0), im1.squeeze(0)))
            im0 = im0.to(device).unsqueeze(0)
            im1 = im1.to(device).unsqueeze(0)
        elif isinstance(im0, torch.Tensor):
            imA, imB = im0, im1
            test_transform = get_tuple_transform_ops(
                resize=(self.h_resized, self.w_resized), normalize=True, clahe = False
            )
            im0, im1 = test_transform((im0.squeeze(0), im1.squeeze(0)))
            batched = False
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            im0 = im0.to(device).unsqueeze(0)
            im1 = im1.to(device).unsqueeze(0)
        B,C,H0,W0 = im0.shape
        B,C,H1,W1 = im1.shape
        self.train(False)
        corresps = self.forward({"im_A":im0, "im_B":im1}, symmetric=self.symmetric)

        if self.upsample_preds:
            hs, ws = self.upsample_res
            ## from coarse to fine
            self.num_grid_up = [int(hs/14), 2*int(hs/14), 4*int(hs/14), 8*int(hs/14)]
            self.radius_up = self.radius[-len(self.num_grid_up):]
            self.num_itr_up = self.num_itr[-len(self.num_grid_up):]
        if self.attenuate_cert:
            low_res_certainty = F.interpolate(
            corresps["16"][self.num_itr[0]]["certainty"], size=(self.num_grid_up[-1], self.num_grid_up[-1]), align_corners=False, mode="bilinear"
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)
        if self.upsample_preds:
            finest_corresps = corresps["1"][self.num_itr[-1]]
            torch.cuda.empty_cache()
            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), mode=2, normalize=True
            )
            im0, im1 = test_transform((imA.squeeze(0), imB.squeeze(0)))
            im0, im1 = im0[None].to(device), im1[None].to(device)
            scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
            batch = {"im_A": im0, "im_B": im1}
            corresps = self.forward(batch, pre_corresps=finest_corresps, scale_factor=scale_factor, upsample=True, symmetric=self.symmetric)

        #return 1,1
        if self.upsample_preds:
            num_grid = self.num_grid_up
            num_itr = self.num_itr_up
        else:
            num_grid = self.num_grid
            num_itr = self.num_itr
        G = num_grid[-1]
        flow = corresps["1"][num_itr[-1]]["flow"].permute(0,2,3,1).reshape(-1,G,G,2)
        certainty = corresps["1"][num_itr[-1]]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
        certainty = certainty.sigmoid()
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1+1/G,1-1/G, G),
                torch.linspace(-1+1/G,1-1/G, G),
                indexing = "xy"),
            dim = -1).float().to(flow.device).expand(B, G, G, 2)
        if (flow.abs() > 1).any() and True:
            wrong = (flow.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:,None]] = 0
        flow = torch.clamp(flow, -1, 1)

        if self.symmetric:
            A_to_B, B_to_A = flow.chunk(2)
            q_warp = torch.cat((grid, A_to_B), dim=-1)
            s_warp = torch.cat((B_to_A, grid), dim=-1)
            warp = torch.cat((q_warp, s_warp),dim=2)
            certainty = torch.cat(certainty.chunk(2), dim=3)
        else:
            warp = torch.cat((grid, flow), dim = -1)
        if batched:
            return warp, certainty[:, 0]
        else:
            return warp[0], certainty[0, 0]
    def sample(
        self,
        matches,
        certainty,
        num=5_000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty,
                        num_samples = min(expansion_factor*num, len(certainty)),
                        replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        use_half = True if matches.device.type == "cuda" else False
        down = 1 if matches.device.type == "cuda" else 8
        density = kde(good_matches, std=0.1, half = use_half, down = down)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p,
                        num_samples = min(num,len(good_certainty)),
                        replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def corr_volume(self, feat0, feat1):
        """
            input:
                feat0 -> torch.Tensor(B, C, H, W)
                feat1 -> torch.Tensor(B, C, H, W)
            return:
                corr_volume -> torch.Tensor(B, H, W, H, W)
        """
        B, C, H0, W0 = feat0.shape
        B, C, H1, W1 = feat1.shape
        feat0 = feat0.view(B, C, H0*W0).contiguous()
        feat1 = feat1.view(B, C, H1*W1).contiguous()
        corr_volume = torch.einsum('bci,bcj->bji', feat0, feat1).reshape(B, H1, W1, H0 , W0).contiguous()/math.sqrt(C) #16*16*16
        return corr_volume

    def pos_embed(self, corr_volume: torch.Tensor):
        B, H1, W1, H0, W0 = corr_volume.shape
        grid = torch.stack(
                torch.meshgrid(
                    torch.linspace(-1+1/W1,1-1/W1, W1),
                    torch.linspace(-1+1/H1,1-1/H1, H1),
                    indexing = "xy"),
                dim = -1).float().to(corr_volume).reshape(H1*W1, 2).contiguous()
        P = corr_volume.reshape(B,H1*W1,H0,W0).contiguous().softmax(dim=1) # B, HW, H, W
        pos_embeddings = torch.einsum('bchw,cd->bdhw', P, grid).contiguous()
        return pos_embeddings



class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_num = None,
        corr_in_other = None,
        no_im_B_fm = False,
        amp = False,
        concat_logits = False,
        use_bias_block_1 = True,
        use_cosine_corr = False,
        disable_local_corr_grad = False,
        is_classifier = False,
        sample_mode = "bilinear",
        norm_type = nn.BatchNorm2d,
        bn_momentum = 0.1,
        amp_dtype = torch.float16,
        use_fm_head=True
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias = use_bias_block_1,
        )

        self.use_fm_head = use_fm_head
        if self.use_fm_head:
            # 使用 FMResidualHead 替换传统的 CNN hidden blocks 和 out_conv
            print("Train with Flow Matching")
            self.fm_head = NewFMResidualHead(
                input_dim=hidden_dim,  # block1 的输出维度
                context_dim=64,
                hidden_dim=64,
                out_dim=out_dim + 1  # 2 for delta_flow, 1 for delta_certainty
            )
        else:
            self.hidden_blocks = nn.Sequential(
                *[
                    self.create_block(
                        hidden_dim,
                        hidden_dim,
                        dw=dw,
                        kernel_size=kernel_size,
                        norm_type=norm_type,
                    )
                    for hb in range(hidden_blocks)
                ]
            )
            self.hidden_blocks = self.hidden_blocks
            self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)

        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_num
        self.local_corr_num = local_corr_num
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias = True,
        norm_type = nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim, momentum = self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels = out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, num_grid, x, y, flow, scale_factor = 1, logits = None):
        b,c,hs,ws = x.shape
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1).contiguous(), align_corners=False, mode = self.sample_mode)
            if self.has_displacement_emb:
                im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / num_grid, 1 - 1 / num_grid, num_grid, device=x.device),
                    torch.linspace(-1 + 1 / num_grid, 1 - 1 / num_grid, num_grid, device=x.device),
                ), indexing='ij'
                )
                im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
                im_A_coords = im_A_coords[None].expand(b, 2, num_grid, num_grid)
                grid_feature = F.grid_sample(x, im_A_coords.permute(0, 2, 3, 1), align_corners=False, mode = self.sample_mode)
                in_displacement = flow-im_A_coords
                emb_in_displacement = self.disp_emb(40/32 * scale_factor * in_displacement)

                # Corr in other means take a kxk grid around the predicted coordinate in other image
                if self.corr_in_other:
                    local_corr = local_correlation((b,c,hs,ws), grid_feature, y, local_radius=self.local_corr_radius, num_grid=num_grid, flow = flow, im_A_coords=None,
                                                    sample_mode = self.sample_mode, grid_based_correlation=False)
                    d = torch.cat((grid_feature, x_hat, emb_in_displacement, local_corr), dim=1)
                else:
                    local_corr = None
                    d = torch.cat((grid_feature, x_hat, emb_in_displacement), dim=1)

            d = self.block1(d)
            if self.use_fm_head:
                steps = 4
                dt = 1.0 / steps
                current_t = 0.0
                x_dis = torch.zeros(d.size(0), 2, d.size(2), d.size(3), device=d.device)
                final_certainty = None
                for i in range(steps):
                    t_input = torch.full((d.size(0),), current_t, device=d.device)
                    v_displacement, v_current_certainty = self.fm_head(d, x_dis, t=t_input)
                    x_dis = x_dis + v_displacement * dt
                    if i == steps - 1:
                        final_certainty = v_current_certainty
                    current_t += dt
                displacement = x_dis
                certainty = final_certainty
            else:
                d = self.hidden_blocks(d)
        if not self.use_fm_head:
            d = self.out_conv(d.float())
            displacement, certainty = d[:, :2], d[:, 2:3]

        return displacement, certainty, local_corr


class ResidualBlock(nn.Module):
    """标准的残差块，带有可选的空洞卷积以扩大感受野"""

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, dilation=1)  # 第二层通常保持标准卷积
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # 如果输入输出维度不匹配，需要映射
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.act(out)
        return out


class FMResidualHead(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim, out_dim=3, time_emb_dim=64):
        """
        升级版 FM Head:
        1. Context Proj: 单层 Conv -> 级联 ResBlock (扩大感受野)
        2. Time Injection: Concat -> FiLM (Scale & Shift)
        """
        super().__init__()

        # --- 改动 1: 增强 Context 编码器 ---
        # 使用两个残差块，第一个使用空洞卷积 (dilation=2) 扩大感受野
        self.context_encoder = nn.Sequential(
            # 先降维/映射到 hidden_dim
            nn.Conv2d(input_dim, context_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            # ResBlock 1: Dilation=2，捕捉稍远的信息
            ResidualBlock(context_dim, hidden_dim, dilation=2),
            # ResBlock 2: 标准卷积，精细化特征
            ResidualBlock(hidden_dim, hidden_dim, dilation=1)
        )

        # --- 改动 2: 增强时间嵌入网络 ---
        self.time_emb_dim = time_emb_dim
        # 时间 MLP 输出维度翻倍 (hidden_dim * 2)，因为 FiLM 需要预测 Scale 和 Shift 两个参数
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # 输出 [Scale, Shift]
        )

        # --- 改动 3: 核心 Flow Network ---
        # 由于我们使用 FiLM 调节特征，这里的输入维度主要由 hidden_dim 决定
        # 加上 Flow 本身的 2 通道
        # 结构：Input(Flow) + Conditioned_Feature -> MLP -> Delta

        self.head_in_conv = nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1)

        self.flow_res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )

        self.final_proj = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def get_time_embedding(self, t, channels):
        """标准的正弦时间嵌入"""
        half_dim = channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        if channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, d, flow):
        B, _, H, W = d.shape

        # 1. 编码 Context 特征 (B, hidden_dim, H, W)
        #    现在这里经过了深层 ResNet，特征表达能力更强
        ctx_feat = self.context_encoder(d)

        # 2. 处理时间嵌入
        t_val = 1.0
        t = torch.full((B,), t_val, device=d.device)
        t_emb = self.get_time_embedding(t, self.time_emb_dim)  # (B, time_emb_dim)

        # 3. FiLM 机制 (Feature-wise Linear Modulation)
        #    预测 Scale (gamma) 和 Shift (beta)
        #    time_style: (B, 2 * hidden_dim)
        time_style = self.time_mlp(t_emb)

        #    Reshape 为 (B, 2*hidden_dim, 1, 1) 以便与特征图广播
        time_style = time_style[:, :, None, None]
        scale, shift = time_style.chunk(2, dim=1)

        #    应用时间调节到 Context 特征上： Out = Scale * Feat + Shift
        #    这比简单的 Concat 能更有效地让时间/状态信息控制特征
        ctx_feat = ctx_feat * (1 + scale) + shift

        # 4. 融合 Flow 信息
        #    先将 flow 映射到 hidden_dim
        flow_feat = self.head_in_conv(flow)

        #    相加融合 (Residual style)
        x = ctx_feat + flow_feat

        # 5. 通过 Flow Network 预测残差
        x = self.flow_res_blocks(x)
        v_pred = self.final_proj(x)

        return v_pred[:, :2], v_pred[:, 2:3]


class NewFMResidualHead(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim, out_dim=3, time_emb_dim=64):
        super().__init__()

        # Context 编码器 (保持不变)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(input_dim, context_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(context_dim, hidden_dim, dilation=2),
            ResidualBlock(hidden_dim, hidden_dim, dilation=1)
        )

        # 时间 MLP (保持不变)
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # Scale, Shift
        )

        # Flow 输入处理
        self.head_in_conv = nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1)

        # --- 改动: 将时间注入机制整合到 ResBlock 或者融合之后 ---
        # 这里为了演示简单，我们在融合 flow 和 context 之后再次应用 FiLM
        # 更好的做法是将 Scale/Shift 传入 ResBlock 内部 (AdaGN)

        self.flow_res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)  # 加深一点
        )

        self.final_proj = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def get_time_embedding(self, t, channels):
        # ... (保持原样) ...
        half_dim = channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        if channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, d, x_t, t):
        """
        d: 条件特征 (Source + Target features)
        x_t: 当前时刻的 flow 估计量 (或者噪声)
        t: 当前时间步 (B,) or scalar, 范围 [0, 1]
        """
        B, _, H, W = d.shape

        # 1. 处理时间 t (必须作为参数传入!)
        if isinstance(t, float) or isinstance(t, int):
            t = torch.full((B,), t, device=d.device)

        # 2. 编码条件
        ctx_feat = self.context_encoder(d)  # (B, hidden, H, W)

        # 3. 处理当前 Flow 状态
        flow_feat = self.head_in_conv(x_t)  # (B, hidden, H, W)

        # 4. 融合
        h = ctx_feat + flow_feat

        # 5. 时间注入 (FiLM) - 作用在融合后的特征上更有效
        t_emb = self.get_time_embedding(t, self.time_emb_dim)
        time_style = self.time_mlp(t_emb)  # (B, 2*hidden)
        scale, shift = time_style[:, :, None, None].chunk(2, dim=1)

        # Apply FiLM: 调节整个融合特征，告诉网络现在是什么阶段
        h = h * (1 + scale) + shift

        # 6. 主干计算
        h = self.flow_res_blocks(h)

        # 7. 预测向量场 v_t
        v_pred = self.final_proj(h)

        return v_pred[:, :2], v_pred[:, 2:3]