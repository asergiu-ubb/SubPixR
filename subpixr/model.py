import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import timm


# SubPixR: iterative sub-pixel refiner. An FPN encoder feeds a template/search
# pair into one of three matching modules — depthwise cross-correlation, a
# local cost volume, or a cross-attention bottleneck (optionally chained as
# attention -> correlation). A spatial soft-argmax head regresses a normalized
# sub-pixel offset. The architecture switches below are kept so older
# checkpoints continue to load.


# --- Spatial soft-argmax ---
class SpatialSoftArgmax(nn.Module):
    def __init__(self, temperature=10.0, grid_bounds=(-8.0, 8.0)):
        super().__init__()
        self.temperature = temperature
        # Store the bounds so forward() doesn't need them passed every time
        self.bounds = grid_bounds

    def forward(self, heatmap):
        # heatmap: [B, C, H, W] -> we only expect 1 channel here
        if heatmap.dim() == 4:
            B, _, H, W = heatmap.shape
            heatmap = heatmap.squeeze(1)  # Drop the channel dim for softmax
        else:
            B, H, W = heatmap.shape

        # Flatten spatial dims to apply Softmax
        # Temperature sharpens the peak so it doesn't just average the whole image
        probs = F.softmax(heatmap.view(B, -1) * self.temperature, dim=1).view(B, H, W)

        # Create normalized X and Y grids based on the stored bounds
        min_val, max_val = self.bounds
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(min_val, max_val, H, device=heatmap.device),
            torch.linspace(min_val, max_val, W, device=heatmap.device),
            indexing='ij'
        )

        # Expected value (Center of Mass)
        expected_x = torch.sum(probs * grid_x.unsqueeze(0), dim=(1, 2))
        expected_y = torch.sum(probs * grid_y.unsqueeze(0), dim=(1, 2))

        return torch.stack([expected_x, expected_y], dim=1)  # [B, 2]


# --- Cross-attention bottleneck ---
class CrossAttentionBottleneck(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, feat_t, feat_s):
        B, C, H, W = feat_t.shape
        seq_t = feat_t.view(B, C, -1).permute(0, 2, 1)
        seq_s = feat_s.view(B, C, -1).permute(0, 2, 1)

        attn_out, _ = self.mha(query=seq_t, key=seq_s, value=seq_s)
        seq_t = self.norm(seq_t + attn_out)

        out_2d = seq_t.permute(0, 2, 1).view(B, C, H, W)
        return out_2d


# --- FPN-style ResNet encoder (stride-4 output) ---
class MultiStageResNetEncoder(nn.Module):

    def __init__(self, encoder_type='resnet18'):
        super().__init__()
        if encoder_type == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # Stride 4, 64 channels
        self.layer2 = resnet.layer2  # Stride 8, 128 channels
        self.layer3 = resnet.layer3  # Stride 16, 256 channels

        # 1x1 Convolutions to squash deep layers to 64 channels to prevent feature dominance
        self.reduce_l2 = nn.Conv2d(128, 64, kernel_size=1)
        self.reduce_l3 = nn.Conv2d(256, 64, kernel_size=1)

    def forward(self, x):
        x_stem = self.stem(x)
        feat1 = self.layer1(x_stem)  # [B, 64, 32, 32]
        feat2 = self.layer2(feat1)  # [B, 128, 16, 16]
        feat3 = self.layer3(feat2)  # [B, 256, 8, 8]

        # Reduce channel depth
        feat2_reduced = self.reduce_l2(feat2)  # [B, 64, 16, 16]
        feat3_reduced = self.reduce_l3(feat3)  # [B, 64, 8, 8]

        # Upsample deep features to match the high-res stride-4 grid of feat1
        feat2_up = F.interpolate(feat2_reduced, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        feat3_up = F.interpolate(feat3_reduced, size=feat1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate: 64 + 64 + 64 = 192 Channels at 32x32 resolution
        return torch.cat([feat1, feat2_up, feat3_up], dim=1)


# --- Main refinement network ---
class RefinementNetwork(nn.Module):
    def __init__(self, encoder_type='resnet18', freeze_encoder=False, dropout_rate=0.4,
                 use_depthwise_xcorr=True, use_attention=False, predict_confidence=False,
                 use_local_cost_volume=False, scale_factor=16.0, use_spatial_head=False,
                 use_multi_stage_features=False, use_pmr_confidence=False,
                 use_hybrid_fusion=False):
        super(RefinementNetwork, self).__init__()
        self.scale_factor = scale_factor
        self.use_spatial_head = use_spatial_head
        self.encoder_type = encoder_type.lower()
        self.freeze_encoder = freeze_encoder
        self.use_depthwise_xcorr = use_depthwise_xcorr
        self.use_attention = use_attention
        self.predict_confidence = predict_confidence
        self.use_local_cost_volume = use_local_cost_volume
        self.use_multi_stage_features = use_multi_stage_features
        self.use_pmr_confidence = use_pmr_confidence
        self.use_hybrid_fusion = use_hybrid_fusion
        self.is_timm = False

        if self.use_pmr_confidence and not self.use_depthwise_xcorr:
            raise ValueError("use_pmr_confidence=True requires use_depthwise_xcorr=True")

        # --- A. Build Encoder (always probed at 128px) ---
        if self.use_multi_stage_features and self.encoder_type in ['resnet18', 'resnet34']:
            self.encoder = MultiStageResNetEncoder(encoder_type=self.encoder_type)
        elif self.encoder_type == 'vgg16':
            vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
            self.encoder = vgg.features[:14]
        elif self.encoder_type == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
                                         resnet.layer2)
        elif self.encoder_type == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
                                         resnet.layer2)
        elif self.encoder_type == 'regnet_y_1_6gf':
            from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights
            regnet = regnet_y_1_6gf(weights=RegNet_Y_1_6GF_Weights.DEFAULT)
            self.encoder = nn.Sequential(regnet.stem, regnet.trunk_output[:2])
        elif self.encoder_type == 'efficientnet_v2_s':
            from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
            effnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            self.encoder = effnet.features[:4]
        else:
            try:
                self.encoder = timm.create_model(
                    self.encoder_type, pretrained=True, features_only=True,
                    out_indices=(2,), img_size=(128, 128)
                )
                self.is_timm = True
            except Exception as e:
                self.encoder = timm.create_model(self.encoder_type, pretrained=True, features_only=True,
                                                 out_indices=(2,))
                self.is_timm = True

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # --- B. Sniff Dimensions ---
        dummy_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            dummy_out = self._extract_features(dummy_input)
            enc_out_channels = dummy_out.shape[1]
            h_f, w_f = dummy_out.shape[2], dummy_out.shape[3]

        if self.use_attention:
            self.attention_bottleneck = CrossAttentionBottleneck(embed_dim=enc_out_channels, num_heads=4)

        # --- C. Channel Math (Backward Compatible) ---
        if not self.use_hybrid_fusion:
            # LEGACY LOGIC (so old checkpoints load)
            if self.use_local_cost_volume:
                in_channels = enc_out_channels + (h_f * w_f)
            elif self.use_attention or not self.use_depthwise_xcorr:
                in_channels = enc_out_channels * 2
            else:
                in_channels = enc_out_channels
        else:
            # NEW HYBRID LOGIC
            if self.use_local_cost_volume:
                in_channels = enc_out_channels + (h_f * w_f)
                if self.use_attention:
                    in_channels += enc_out_channels  # append attended_t
            elif self.use_depthwise_xcorr:
                in_channels = enc_out_channels  # Just corr_volume (no feat_s)
                if self.use_attention:
                    in_channels += enc_out_channels  # append cropped attended_t
            elif self.use_attention or not self.use_depthwise_xcorr:
                in_channels = enc_out_channels * 2

        # PMR confidence appends its channel at runtime — regressor always outputs 2
        out_features = 3 if (self.predict_confidence and not self.use_pmr_confidence) else 2

        # --- D. Regression Heads ---
        if self.use_spatial_head:
            self.spatial_convs = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.heatmap_proj = nn.Conv2d(128, 1, kernel_size=1)
            self.soft_argmax = SpatialSoftArgmax(temperature=10.0, grid_bounds=(-4.0, 4.0))

            if self.predict_confidence and not self.use_pmr_confidence:
                self.conf_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, 1)
                )
        else:
            self.regressor = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(64, out_features)
            )

    def _extract_features(self, x):
        feat = self.encoder(x)
        return feat[-1] if isinstance(feat, (list, tuple)) else feat

    def xcorr_depthwise(self, search, kernel):
        batch, channel, h_s, w_s = search.shape
        _, _, h_k, w_k = kernel.shape
        search = search.reshape(1, batch * channel, h_s, w_s)
        kernel = kernel.reshape(batch * channel, 1, h_k, w_k)
        out = F.conv2d(search, kernel, groups=batch * channel)
        return out.reshape(batch, channel, out.size(2), out.size(3))

    def extract_patch(self, img_or_feat, coords, patch_size):
        B, C, H, W = img_or_feat.shape
        device = img_or_feat.device

        t = torch.linspace(-1, 1, patch_size, device=device)
        mesh_y, mesh_x = torch.meshgrid(t, t, indexing='ij')
        grid = torch.stack((mesh_x, mesh_y), 2).unsqueeze(0).repeat(B, 1, 1, 1)

        x0 = ((coords[:, 0:1] / W) * 2 - 1).view(B, 1, 1)
        x1 = (((coords[:, 0:1] + patch_size) / W) * 2 - 1).view(B, 1, 1)
        y0 = ((coords[:, 1:2] / H) * 2 - 1).view(B, 1, 1)
        y1 = (((coords[:, 1:2] + patch_size) / H) * 2 - 1).view(B, 1, 1)

        grid_new = torch.empty_like(grid)
        grid_new[:, :, :, 0] = grid[:, :, :, 0] * (x1 - x0) / 2 + (x1 + x0) / 2
        grid_new[:, :, :, 1] = grid[:, :, :, 1] * (y1 - y0) / 2 + (y1 + y0) / 2

        return F.grid_sample(img_or_feat, grid_new, align_corners=False, mode='bilinear')

    def forward(self, template, search, num_iters=1, visual_debug=False, return_intermediate=False,
                return_logits=False):
        B = template.shape[0]
        _, _, h_s, w_s = search.shape
        feat_t = self._extract_features(template)

        # Multi-iter logic
        if num_iters > 1 and (h_s > 128 or w_s > 128):
            cx = torch.full((B,), w_s / 2.0, device=search.device)
            cy = torch.full((B,), h_s / 2.0, device=search.device)
            accumulated_delta = torch.zeros((B, 2), device=search.device)
            debug_patches = []
            intermediate_deltas = []

            for _ in range(num_iters):
                cx = cx.clamp(64.0, w_s - 64.0)
                cy = cy.clamp(64.0, h_s - 64.0)
                top_left = torch.stack([cx - 64.0, cy - 64.0], dim=1)

                if visual_debug:
                    debug_patches.append(self._add_crosshair(self.extract_patch(search, top_left, 128)))

                crop = self.extract_patch(search, top_left, 128)
                feat_s = self._extract_features(crop)
                out = self._fuse_and_regress(feat_t, feat_s)

                raw_step_delta = out[:, :2]

                # --- Gate the delta ONLY at inference ---
                if (self.predict_confidence or self.use_pmr_confidence) and not self.training:
                    conf = out[:, 2:3] if self.use_pmr_confidence else torch.sigmoid(out[:, 2:3])
                    step_delta = raw_step_delta * conf
                else:
                    step_delta = raw_step_delta

                accumulated_delta = accumulated_delta + step_delta

                # Detach coordinates, move using the GATED delta so the crop matches the output
                cx = (cx + step_delta[:, 0] * self.scale_factor).detach()
                cy = (cy + step_delta[:, 1] * self.scale_factor).detach()

                if return_intermediate:
                    intermediate_deltas.append(accumulated_delta)

            if visual_debug:
                return accumulated_delta, debug_patches

            if return_intermediate:
                # PMR has no conf loss — only learned confidence needs raw_out
                if self.training and self.predict_confidence:
                    return intermediate_deltas, out
                return intermediate_deltas

            if (self.training or return_logits) and self.predict_confidence:
                return out

            return accumulated_delta

        # Single-shot logic
        else:
            feat_s = self._extract_features(search)
            out = self._fuse_and_regress(feat_t, feat_s)

            # Return raw [B, 3] during training or when logits requested.
            # PMR: prevents confidence gating from distorting the training loss.
            if (self.training or return_logits) and (self.predict_confidence or self.use_pmr_confidence):
                return out

            if self.predict_confidence or self.use_pmr_confidence:
                delta = out[:, :2]
                conf = out[:, 2:3] if self.use_pmr_confidence else torch.sigmoid(out[:, 2:3])
                final_delta = delta * conf
            else:
                final_delta = out[:, :2]

            if visual_debug:
                return final_delta, [self._add_crosshair(search)]
            return final_delta

    def _fuse_and_regress(self, feat_t, feat_s):
        if not self.use_hybrid_fusion:
            # Legacy fusion: attention and correlation are independent branches.
            if self.use_local_cost_volume:
                b, c, h, w = feat_s.shape
                f_t_n = F.normalize(feat_t, p=2, dim=1)
                f_s_n = F.normalize(feat_s, p=2, dim=1)
                corr = torch.bmm(f_t_n.view(b, c, -1).transpose(1, 2), f_s_n.view(b, c, -1))
                fused_feat = torch.cat([feat_s, corr.view(b, h * w, h, w)], dim=1)
            elif self.use_attention:
                attended_t = self.attention_bottleneck(feat_t, feat_s)
                fused_feat = torch.cat((attended_t, feat_s), dim=1)
            elif self.use_depthwise_xcorr:
                h, w = feat_t.shape[2], feat_t.shape[3]
                kh, kw = h // 4, w // 4
                fused_feat = self.xcorr_depthwise(feat_s, feat_t[:, :, kh:h - kh, kw:w - kw])
            else:
                fused_feat = torch.cat((feat_t, feat_s), dim=1)

            # Legacy PMR
            if self.use_pmr_confidence and self.use_depthwise_xcorr:
                with torch.no_grad():
                    corr_map = fused_feat.mean(dim=1)
                    flat = corr_map.view(corr_map.shape[0], -1)
                    peak = flat.max(dim=1)[0]
                    mean = flat.mean(dim=1)
                    pmr = peak / (mean + 1e-6)
                    auto_conf = torch.sigmoid(pmr - 2.0).unsqueeze(1)
        else:
            # Hybrid fusion: template is passed through attention first, then
            # the attended template feeds the correlation/cost branch.
            if self.use_attention:
                attended_t = self.attention_bottleneck(feat_t, feat_s)
                feat_t_active = attended_t
            else:
                feat_t_active = feat_t
                attended_t = None

            if self.use_local_cost_volume:
                b, c, h, w = feat_s.shape
                f_t_n = F.normalize(feat_t_active, p=2, dim=1)
                f_s_n = F.normalize(feat_s, p=2, dim=1)
                corr = torch.bmm(f_t_n.view(b, c, -1).transpose(1, 2), f_s_n.view(b, c, -1))
                corr_volume = corr.view(b, h * w, h, w)

                fused_list = [feat_s, corr_volume]
                if self.use_attention:
                    fused_list.append(attended_t)

            elif self.use_depthwise_xcorr:
                h, w = feat_t_active.shape[2], feat_t_active.shape[3]
                kh, kw = h // 4, w // 4
                corr_volume = self.xcorr_depthwise(feat_s, feat_t_active[:, :, kh:h - kh, kw:w - kw])

                # We drop feat_s here entirely to match legacy xcorr behavior.
                fused_list = [corr_volume]

                if self.use_attention:
                    # Crop attended_t (32x32) down to exactly match corr_volume (17x17)
                    ch, cw = corr_volume.shape[2], corr_volume.shape[3]
                    pad_h = (h - ch) // 2
                    pad_w = (w - cw) // 2
                    attended_t_cropped = attended_t[:, :, pad_h:pad_h + ch, pad_w:pad_w + cw]
                    fused_list.append(attended_t_cropped)
            else:
                fused_list = [feat_t_active, feat_s]

            fused_feat = torch.cat(fused_list, dim=1)

            # Hybrid PMR (pulls from isolated corr_volume)
            if self.use_pmr_confidence and (self.use_depthwise_xcorr or self.use_local_cost_volume):
                with torch.no_grad():
                    corr_map = corr_volume.mean(dim=1)
                    flat = corr_map.view(corr_map.shape[0], -1)
                    peak = flat.max(dim=1)[0]
                    mean = flat.mean(dim=1)
                    pmr = peak / (mean + 1e-6)
                    auto_conf = torch.sigmoid(pmr - 2.0).unsqueeze(1)

        # --- Common Regression Logic ---
        if self.use_spatial_head:
            hidden = self.spatial_convs(fused_feat)
            heatmap = self.heatmap_proj(hidden)
            delta = self.soft_argmax(heatmap)

            if self.use_pmr_confidence:
                return torch.cat([delta, auto_conf], dim=1)
            elif self.predict_confidence:
                # DETACH hidden so BCE loss doesn't backprop into the spatial convs/encoder
                return torch.cat([delta, self.conf_head(hidden.detach())], dim=1)
            return delta
        else:
            out = self.regressor(fused_feat)
            if self.use_pmr_confidence:
                return torch.cat([out[:, :2], auto_conf], dim=1)
            return out

    def _add_crosshair(self, patch):
        p = patch.clone()
        p[:, 0, 62:66, 62:66] = 1.0
        p[:, 1, 62:66, 62:66] = 0.0
        p[:, 2, 62:66, 62:66] = 0.0
        return p