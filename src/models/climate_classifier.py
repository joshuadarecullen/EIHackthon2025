import torch
from .met_module import MetLitModule


class ClimateClassifier(torch.nn.Module):
    def __init__(self,
                 encoder: str,
                 num_classes: int,
                 output_dim: int,
                 encoder_freeze: bool):
        super().__init__()
        self.model = MetLitModule.load_from_checkpoint(encoder)
        # self.encoder = encoder  # from your trained CLIP model
        self.classifier = torch.nn.Linear(512, num_classes)

        # if encoder_freeze == False:
        #     for param in self.model.parameters():
        #         param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():  # optional: freeze encoder
            features = self.model.net.encode_climate_data(x)
            print(features.shape)
        return self.classifier(features)
