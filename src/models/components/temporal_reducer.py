import torch
import torch.nn as nn

"""
Cons:

    You lose temporal information (e.g., movement of weather systems).

    Not ideal if time evolution is important to interpreting the scene.

    May make descriptions like "a cold front is approaching" harder to align with the image.

time averaging if:

    The text descriptions summarise the entire sequence, not a single timestep.

    You're doing a proof of concept or want to fine-tune a small model quickly.

    You want to reuse off-the-shelf CLIP encoders without modifying them.
"""
class TemporalReducer(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 time_dim=8,):

        super().__init__()
        # Input: [3, 8, 120, 160] → view as [1, 3, 8, 120, 160]
        # Conv3d to reduce time dim to 1
        self.reduce_time = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(time_dim, 1, 1),
            stride=(1, 1, 1),
            groups=3, # process information separately
        )

    def forward(self, x):
        # x: [B, 3, 8, 120, 160]
        x = self.reduce_time(x)  # → [B, 3, 1, 120, 160]
        x = x.squeeze(2)         # → [B, 3, 120, 160]
        return x


# self.reduce_time = nn.Sequential(
#     nn.Conv3d(...),
#     nn.ReLU(),
#     nn.BatchNorm3d(...)
# )

