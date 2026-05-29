from .pretrained.spikenet_features import spikenet_features, spikenet_features_summary
from .skeleton import EmbeddedBackbone

base_architecture_to_features = {
    "spikenet": spikenet_features,
    "spikenet_summary": spikenet_features_summary,
}


def construct_backbone(base_architecture, pretrained=True):
    if "spikenet" in base_architecture:
        return EmbeddedBackbone(
            base_architecture_to_features[base_architecture](pretrained=pretrained),
            input_channels=(1, 128, 37),
        )

    return EmbeddedBackbone(
        base_architecture_to_features[base_architecture](pretrained=pretrained)
    )
from .pretrained.spikenet_features import spikenet_features, spikenet_features_summary
from .skeleton import EmbeddedBackbone

base_architecture_to_features = {
    "spikenet": spikenet_features,
    "spikenet_summary": spikenet_features_summary,
}


def construct_backbone(base_architecture, pretrained=True):
    if "spikenet" in base_architecture:
        return EmbeddedBackbone(
            base_architecture_to_features[base_architecture](pretrained=pretrained),
            input_channels=(1, 128, 37),
        )

    return EmbeddedBackbone(
        base_architecture_to_features[base_architecture](pretrained=pretrained)
    )
