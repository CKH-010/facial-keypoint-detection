"""NaimishNet CNN (Agarwal et al., 2017) for facial keypoint detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""BAM(Bottleneck Attention Module), and CoordConv."""
class ChannelAttention(nn.Module):
    """
    Bottleneck Attention Module (Channel Attention) -> squashes 256 channel to 16 channel and back.
    This is a tiny MLP for channel attention.
    The purpose is showing types of details (channels) that actually matter.
    """
    
    def __init__(self, channels=256, reduction=16):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), #256 channel to 16 channel **Linear = 2D MLP
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False), #16 channel back to 256 channel
        )

    def forward(self, x):
        b, c, _, _ = x.size() # batch size, channels, height, width
        
        # Global Average pool(To look every details in the feature map) to 1x1 spatial dimensions. Giving the global context for each channel.
        # 10X10 grid would be reduced to 1X1 by adaptive_avg_pool2d basically blends all the spatial pixels together within each channel.
        # purpose is to capture the generally important information for each channel.
        avg_pool = F.adaptive_avg_pool2d(x, 1) 

        #This basically flatten from 4D to 2D, then run the tiny MLP for channel attention(bottleneck), then reshape back to 4D.

        """
        Basically, this is the implementation of the channel attention mechanism from the return function 
        
        flat_pool = avg_pool.squeeze(-1).squeeze(-1) # Flatten: [B, C, 1, 1] -> [B, C]  for mlp 

        bottleneck_out = self.fc(flat_pool) # Run MLP

        return bottleneck_out.unsqueeze(-1).unsqueeze(-1) # Reshape: [B, C] -> [B, C, 1, 1]
        """

        return self.fc(avg_pool.view(b, c)).view(b, c, 1, 1)



class spatial_attention(nn.Module):

    def __init__(self, channels=256, reduction=16, dilation=4):
        """
        Bottleneck Attention Module (Spatial Attention) -> captures spatially important regions.
        Just find where are the important spatial regions in the feature map by squashing the channel dimension and focusing on spatial locations.
        """
        super().__init__()
        hidden_channels = channels // reduction
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),  # Squash the channels from 256 down to 16. 
            nn.BatchNorm2d(hidden_channels), # Normalization
            nn.ReLU(inplace=True),
            # we look at 3x3 area before. With dilation, Now we are looking at 9x9 receptive field.
            # This helps the network to capture larger context without increasing the kernel size.
            # This lets the model see the "big picture" on a tiny 10x10 grid.
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                      padding=dilation, dilation=dilation),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1), # Collapse all 16 channels down to just 1 channel.
        )

    def forward(self, x):
        return self.conv(x)


class BAM(nn.Module): # this just combines channel and spatial attention mechanisms
    def __init__(self, channels=256, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = spatial_attention(channels, reduction)

    def forward(self, x):
        attention = self.channel_attention(x) + self.spatial_attention(x)
        return x * torch.sigmoid(attention)

class CoordConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.size()
        # we convert the the image with coordinate system so we can see -1 to 1 with x and y coordinates
        x_lin = torch.linspace(-1, 1, w, device=x.device)  # Create a linear space for x coordinates
        y_lin = torch.linspace(-1, 1, h, device=x.device)  # Create a linear space for y coordinates

        ## We put y first because 'ij' means [row (y), column (x)].Matrix notation LOL
        # Create a meshgrid for y and x coordinates , basically define the coordinate for each pixel in the image
        y_grid, x_grid = torch.meshgrid(y_lin, x_lin, indexing='ij')  

        # Stack along a new dimension to create a 2-channel coordinate map. Shape: [2, 96, 96] where it represent [dimention, height, width]
        grid = torch.stack([x_grid, y_grid], dim=0) 

        # Unsqueeze to add a batch dimension and expand to match the input batch size. This avoids repeating the full grid for each batch element, saving memory.
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

        return torch.cat([x, grid], dim=1)# mark the input with coordinate channels


class NaimishNet(nn.Module):
    """Four conv blocks + three FC layers for 96x96 grayscale input.

    The final layer outputs 180 values (90 keypoints x (x, y)).
    """

    def __init__(self, num_outputs: int = 180, use_bam_and_coord: bool = False):
        super().__init__()
        self.use_advanced = use_bam_and_coord
        in_channels = 3 if self.use_advanced else 1
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=2), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=1), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1000), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(1000, 1000), nn.Linear(1000, num_outputs),
        )

        if use_bam_and_coord:
            self.use_advanced = True
            self.bam = BAM()  # Assuming BAM is defined elsewhere
            self.coord = CoordConv()  # Assuming CoordConv is defined elsewhere
        else:
            self.use_advanced = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_advanced:
            x = self.coord(x)
            
        x = self.features(x)
        if self.use_advanced:
            x = self.bam(x)
        
        return self.classifier(x)
