import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


DEFAULT_VIT_CHECKPOINT = (
    "xxx/"
    "pretrained_models/jx_vit_large_p16_224-4ee7a4dc.pth"
)



def load_vit_backbone(checkpoint_path):
    """Load ViT-L/16 without its classification head."""
    backbone = create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]

    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in checkpoint.items()
        if not key.startswith("head.")
    }
    backbone.load_state_dict(state_dict, strict=False)
    return backbone


class SharedVisualEncoder(nn.Module):
    """Shared ViT encoder for person, union, and full-image inputs."""

    def __init__(self, backbone, feature_dim=512, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone
        self.backbone_dim = backbone.num_features

        self.projection = nn.Linear(
            self.backbone_dim,
            feature_dim,
            bias=False,
        )
        self.norm = nn.LayerNorm(feature_dim)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _extract_tokens(self, images):
        if self.freeze_backbone:
            with torch.no_grad():
                tokens = self.backbone.forward_features(images)
        else:
            tokens = self.backbone.forward_features(images)

        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.ndim != 3:
            raise RuntimeError(
                "The ViT backbone must return [B, tokens, channels]."
            )
        return tokens

    def _project(self, features):
        shape = features.shape
        features = features.reshape(-1, shape[-1]).float()
        features = self.norm(self.projection(features))
        return features.reshape(*shape[:-1], self.feature_dim)

    def forward(self, images, return_patch_tokens=False):
        leading_shape = images.shape[:-3]
        images = images.reshape(-1, *images.shape[-3:])
        tokens = self._extract_tokens(images)

        if tokens.shape[1] == 197:
            global_tokens = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        else:
            patch_tokens = tokens
            global_tokens = tokens.mean(dim=1)

        global_tokens = self._project(global_tokens)
        global_tokens = global_tokens.reshape(*leading_shape, self.feature_dim)

        if not return_patch_tokens:
            return global_tokens

        patch_tokens = self._project(patch_tokens)
        patch_tokens = patch_tokens.reshape(
            *leading_shape,
            patch_tokens.shape[1],
            self.feature_dim,
        )
        return global_tokens, patch_tokens


class MultiScaleTokenEncoder(nn.Module):
    """Build ordered 1x1, 2x2, and 4x4 window tokens."""

    def __init__(self, feature_dim=512, scales=(1, 2, 4), base_grid=14):
        super().__init__()
        self.feature_dim = feature_dim
        self.scales = tuple(scales)
        self.base_grid = base_grid
        self.token_lengths = tuple(
            (base_grid - scale + 1) ** 2 for scale in self.scales
        )

        self.window_projections = nn.ModuleDict()
        self.position_embeddings = nn.ParameterDict()
        self.scale_embeddings = nn.ParameterDict()

        for scale, token_length in zip(self.scales, self.token_lengths):
            if scale == 1:
                self.window_projections[str(scale)] = nn.Identity()
            else:
                self.window_projections[str(scale)] = nn.Sequential(
                    nn.Linear(
                        feature_dim * scale * scale,
                        feature_dim,
                        bias=False,
                    ),
                    nn.LayerNorm(feature_dim),
                )

            self.position_embeddings[str(scale)] = nn.Parameter(
                torch.zeros(1, token_length, feature_dim)
            )
            self.scale_embeddings[str(scale)] = nn.Parameter(
                torch.zeros(1, 1, feature_dim)
            )
            nn.init.normal_(
                self.position_embeddings[str(scale)],
                std=0.02,
            )
            nn.init.normal_(
                self.scale_embeddings[str(scale)],
                std=0.02,
            )

    def forward(self, patch_tokens):
        batch_size, token_count, feature_dim = patch_tokens.shape
        expected = self.base_grid * self.base_grid
        if token_count != expected:
            raise ValueError(
                f"Expected {expected} patch tokens, received {token_count}."
            )

        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            feature_dim,
            self.base_grid,
            self.base_grid,
        )

        multi_scale_tokens = []
        for scale in self.scales:
            windows = F.unfold(
                feature_map,
                kernel_size=scale,
                stride=1,
            ).transpose(1, 2)
            tokens = self.window_projections[str(scale)](windows)
            tokens = (
                tokens
                + self.position_embeddings[str(scale)]
                + self.scale_embeddings[str(scale)]
            )
            multi_scale_tokens.append(tokens)

        # The order is fixed: 196 1x1 tokens, 169 2x2 tokens, 121 4x4 tokens.
        return torch.cat(multi_scale_tokens, dim=1)


class SpatialConfigEncoder(nn.Module):
    def __init__(self, spatial_dim=6, geometry_dim=32):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(spatial_dim, geometry_dim, bias=False),
            nn.LayerNorm(geometry_dim),
        )

    def forward(self, spatial_config):
        return self.projection(torch.nan_to_num(spatial_config.float()))


class PairQueryEncoder(nn.Module):
    """Construct a query for one target person pair."""

    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
        )

    def forward(
        self,
        person1_feature,
        person2_feature,
        union_feature,
        spatial_feature,
    ):
        pair_feature = torch.cat(
            [
                person1_feature + person2_feature,
                torch.abs(person1_feature - person2_feature),
                union_feature,
                spatial_feature,
            ],
            dim=-1,
        )
        return self.projection(pair_feature)


class PairConditionedCrossAttention(nn.Module):
    """Use a pair query to retrieve relation evidence from image tokens."""

    def __init__(self, feature_dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.spatial_bias = nn.Linear(
            2,
            num_heads,
            bias=False,
        )
        nn.init.zeros_(self.spatial_bias.weight)

    def forward(self, pair_query, image_tokens, spatial_token_features):
        attention_mask = None
        if spatial_token_features is not None:
            batch_size, token_count, _ = spatial_token_features.shape
            bias = self.spatial_bias(spatial_token_features)
            attention_mask = bias.permute(0, 2, 1).reshape(
                batch_size * self.num_heads,
                1,
                token_count,
            )

        attended, attention = self.attention(
            pair_query.unsqueeze(1),
            image_tokens,
            image_tokens,
            attn_mask=attention_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        # attention: [R, heads, 1, 486] -> [R, 486]
        attention = attention[:, :, 0, :].mean(dim=1)
        return attended[:, 0], attention


class network(nn.Module):
    """Pair-conditioned multi-scale visual relation recognition network.

    Forward output:
        logits: [R, num_classes]
        visualization_info: dictionary containing four maps, each [R, 1, 224, 224]:
            heatmap_1x1, heatmap_2x2, heatmap_4x4, heatmap_fused.
    """

    def __init__(
        self,
        num_classes=6,
        spatial_dim=6,
        vit_checkpoint=DEFAULT_VIT_CHECKPOINT,
        feature_dim=512,
        num_heads=8,
        freeze_backbone=True,
        geometry_dim=32,
    ):
        super().__init__()

        visual_backbone = load_vit_backbone(vit_checkpoint)
        self.visual_encoder = SharedVisualEncoder(
            backbone=visual_backbone,
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone,
        )

        self.spatial_encoder = SpatialConfigEncoder(
            spatial_dim=spatial_dim,
            geometry_dim=geometry_dim,
        )
        self.pair_query_encoder = PairQueryEncoder(
            input_dim=feature_dim * 3 + geometry_dim,
            feature_dim=feature_dim,
        )
        self.multi_scale_encoder = MultiScaleTokenEncoder(
            feature_dim=feature_dim,
            scales=(1, 2, 4),
            base_grid=14,
        )
        self.cross_attention = PairConditionedCrossAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                feature_dim * 3 + geometry_dim,
                feature_dim,
                bias=False,
            ),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                feature_dim,
                num_classes,
            ),
        )

    @staticmethod
    def _expand_full_image_features(
        full_image_features,
        image_relation_counts,
        relation_count,
    ):
        if image_relation_counts is None:
            if full_image_features.shape[0] != relation_count:
                raise ValueError(
                    "image_relation_counts is required when the number of "
                    "full images differs from the number of person pairs."
                )
            return full_image_features

        counts = image_relation_counts.reshape(-1).long()
        if int(counts.sum().item()) != relation_count:
            raise ValueError(
                "The sum of image_relation_counts must equal the number of "
                "person pairs."
            )

        return full_image_features.repeat_interleave(
            counts.to(full_image_features.device),
            dim=0,
        )

    @staticmethod
    def _build_spatial_token_features(
        pair_mask,
        interaction_mask,
        scales=(1, 2, 4),
        base_grid=14,
    ):
        pair_grid = F.interpolate(
            pair_mask.float(),
            size=(base_grid, base_grid),
            mode="area",
        )
        interaction_grid = F.interpolate(
            interaction_mask.float(),
            size=(base_grid, base_grid),
            mode="area",
        )

        spatial_features = []
        for scale in scales:
            pair_windows = F.avg_pool2d(
                pair_grid,
                kernel_size=scale,
                stride=1,
            ).flatten(1)
            interaction_windows = F.avg_pool2d(
                interaction_grid,
                kernel_size=scale,
                stride=1,
            ).flatten(1)
            spatial_features.append(
                torch.stack(
                    [pair_windows, interaction_windows],
                    dim=-1,
                )
            )

        return torch.cat(spatial_features, dim=1)

    def _window_attention_to_patch_map(self, attention, scale):
        relation_count = attention.shape[0]
        window_count = (self.multi_scale_encoder.base_grid - scale + 1) ** 2
        attention = attention.reshape(relation_count, 1, window_count)

        # Distribute each window weight over the base patches it covers.
        values = attention.expand(
            -1,
            scale * scale,
            -1,
        ).contiguous()
        patch_sum = F.fold(
            values,
            output_size=(self.multi_scale_encoder.base_grid,) * 2,
            kernel_size=scale,
            stride=1,
        )
        patch_count = F.fold(
            torch.ones_like(values),
            output_size=(self.multi_scale_encoder.base_grid,) * 2,
            kernel_size=scale,
            stride=1,
        )
        patch_map = patch_sum / patch_count.clamp_min(1.0)
        return patch_map / patch_map.sum(
            dim=(1, 2, 3),
            keepdim=True,
        ).clamp_min(1e-6)

    @staticmethod
    def _resize_for_visualization(patch_map):
        heatmap = F.interpolate(
            patch_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        heatmap = heatmap - heatmap.amin(
            dim=(2, 3),
            keepdim=True,
        )
        return heatmap / heatmap.amax(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(1e-6)

    def _build_visualization_info(self, token_attention):
        scale_attention = torch.split(
            token_attention,
            self.multi_scale_encoder.token_lengths,
            dim=1,
        )

        patch_maps = [
            self._window_attention_to_patch_map(attention, scale)
            for attention, scale in zip(
                scale_attention,
                self.multi_scale_encoder.scales,
            )
        ]

        scale_mass = torch.stack(
            [attention.sum(dim=1) for attention in scale_attention],
            dim=1,
        )
        scale_mass = scale_mass / scale_mass.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        fused_patch_map = sum(
            mass[:, None, None, None] * patch_map
            for mass, patch_map in zip(
                scale_mass.unbind(dim=1),
                patch_maps,
            )
        )

        heatmaps = [
            self._resize_for_visualization(patch_map)
            for patch_map in patch_maps
        ]
        heatmaps.append(self._resize_for_visualization(fused_patch_map))

        return {
            "heatmap_1x1": heatmaps[0],
            "heatmap_2x2": heatmaps[1],
            "heatmap_4x4": heatmaps[2],
            "heatmap_fused": heatmaps[3],
        }

    def _encode_pair_features(
            self,
            union_crop,
            person1_crop,
            person2_crop,
            spatial_config,
    ):
        crop_features = self.visual_encoder(
            torch.cat(
                [union_crop, person1_crop, person2_crop],
                dim=0,
            )
        )

        union_feature, person1_feature, person2_feature = (
            crop_features.chunk(3, dim=0)
        )

        spatial_feature = self.spatial_encoder(spatial_config)

        pair_query = self.pair_query_encoder(
            person1_feature=person1_feature,
            person2_feature=person2_feature,
            union_feature=union_feature,
            spatial_feature=spatial_feature,
        )

        return union_feature, spatial_feature, pair_query

    def _classify_tokens(
            self,
            pair_query,
            union_feature,
            spatial_feature,
            multi_scale_tokens,
            spatial_token_features,
    ):
        attended_feature, token_attention = self.cross_attention(
            pair_query,
            multi_scale_tokens,
            spatial_token_features,
        )

        classifier_input = torch.cat(
            [
                attended_feature,
                pair_query,
                union_feature,
                spatial_feature,
            ],
            dim=-1,
        )

        logits = self.classifier(classifier_input)

        return logits, token_attention

    @staticmethod
    def _build_intervention_masks(
            heatmap_fused,
            top_k,
    ):
        patch_scores = F.adaptive_avg_pool2d(
            heatmap_fused.detach(),
            output_size=(14, 14),
        ).flatten(1)

        top_k = min(
            int(top_k),
            patch_scores.shape[1],
        )

        top_indices = patch_scores.topk(
            top_k,
            dim=1,
            largest=True,
        ).indices

        low_indices = patch_scores.topk(
            top_k,
            dim=1,
            largest=False,
        ).indices

        top_mask = torch.zeros_like(patch_scores)
        low_mask = torch.zeros_like(patch_scores)

        top_mask.scatter_(1, top_indices, 1.0)
        low_mask.scatter_(1, low_indices, 1.0)

        return (
            top_mask.unsqueeze(-1),
            low_mask.unsqueeze(-1),
        )

    def forward_with_interventions(
            self,
            full_image,
            union_crop,
            person1_crop,
            person2_crop,
            pair_mask,
            interaction_mask,
            spatial_config,
            image_relation_counts=None,
            top_k=10,
    ):
        relation_count = union_crop.shape[0]

        union_feature, spatial_feature, pair_query = (
            self._encode_pair_features(
                union_crop,
                person1_crop,
                person2_crop,
                spatial_config,
            )
        )

        _, full_patch_tokens = self.visual_encoder(
            full_image,
            return_patch_tokens=True,
        )

        full_patch_tokens = self._expand_full_image_features(
            full_patch_tokens,
            image_relation_counts=image_relation_counts,
            relation_count=relation_count,
        )

        spatial_token_features = (
            self._build_spatial_token_features(
                pair_mask,
                interaction_mask,
            )
        )

        original_tokens = self.multi_scale_encoder(
            full_patch_tokens
        )

        logits, token_attention = self._classify_tokens(
            pair_query,
            union_feature,
            spatial_feature,
            original_tokens,
            spatial_token_features,
        )

        visualization_info = (
            self._build_visualization_info(
                token_attention
            )
        )

        top_mask, low_mask = (
            self._build_intervention_masks(
                visualization_info["heatmap_fused"],
                top_k,
            )
        )

        top_mask = top_mask.to(
            full_patch_tokens.dtype
        )
        low_mask = low_mask.to(
            full_patch_tokens.dtype
        )

        patch_tokens_keep = (
                full_patch_tokens * top_mask
        )

        patch_tokens_drop = (
                full_patch_tokens * (1.0 - top_mask)
        )

        patch_tokens_low = (
                full_patch_tokens * (1.0 - low_mask)
        )

        intervention_patch_tokens = torch.cat(
            [
                patch_tokens_keep,
                patch_tokens_drop,
                patch_tokens_low,
            ],
            dim=0,
        )

        intervention_tokens = (
            self.multi_scale_encoder(
                intervention_patch_tokens
            )
        )

        intervention_logits, _ = (
            self._classify_tokens(
                pair_query.repeat(3, 1),
                union_feature.repeat(3, 1),
                spatial_feature.repeat(3, 1),
                intervention_tokens,
                spatial_token_features.repeat(
                    3,
                    1,
                    1,
                ),
            )
        )

        logits_keep, logits_drop, logits_low = (
            intervention_logits.chunk(3, dim=0)
        )

        return (
            logits,
            visualization_info,
            logits_keep,
            logits_drop,
            logits_low,
        )

    def forward(
        self,
        full_image,
        union_crop,
        person1_crop,
        person2_crop,
        pair_mask,
        interaction_mask,
        spatial_config,
        image_relation_counts=None,
    ):
        relation_count = union_crop.shape[0]

        crop_features = self.visual_encoder(
            torch.cat(
                [union_crop, person1_crop, person2_crop],
                dim=0,
            )
        )
        union_feature, person1_feature, person2_feature = crop_features.chunk(
            3,
            dim=0,
        )

        spatial_feature = self.spatial_encoder(spatial_config)
        pair_query = self.pair_query_encoder(
            person1_feature=person1_feature,
            person2_feature=person2_feature,
            union_feature=union_feature,
            spatial_feature=spatial_feature,
        )

        _, full_patch_tokens = self.visual_encoder(
            full_image,
            return_patch_tokens=True,
        )
        full_patch_tokens = self._expand_full_image_features(
            full_patch_tokens,
            image_relation_counts=image_relation_counts,
            relation_count=relation_count,
        )

        multi_scale_tokens = self.multi_scale_encoder(full_patch_tokens)
        spatial_token_features = self._build_spatial_token_features(
            pair_mask,
            interaction_mask,
        )

        attended_feature, token_attention = self.cross_attention(
            pair_query,
            multi_scale_tokens,
            spatial_token_features,
        )

        classifier_input = torch.cat(
            [
                attended_feature,
                pair_query,
                union_feature,
                spatial_feature,
            ],
            dim=-1,
        )
        logits = self.classifier(classifier_input)

        visualization_info = self._build_visualization_info(token_attention)
        return logits, visualization_info
