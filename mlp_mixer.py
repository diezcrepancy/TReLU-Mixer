import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from einops.layers.torch import Rearrange
import torchsummary

def set_seed(seed: int):
    
    # Set PyTorch's random seed for CPU and CUDA
    torch.manual_seed(seed)
    
    # If using CUDA (GPU), set additional seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Example usage
SEED = 2511
set_seed(SEED)

# Constraint classes
class NonNegConstraint(nn.Module):
    def forward(self, x):
        return x.clamp(min=0)  # Ensure values are >= 0

class MinMaxConstraint(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        
    def forward(self, x):
        return x.clamp(min=self.min_val, max=self.max_val)

class TReLU(nn.Module):
    def __init__(self, l2_reg=0.000001, initial_r1=None, initial_r2=1.0):
        super().__init__()
        self.l2_reg = l2_reg
        
        # Initialize r1 with normal distribution if not specified
        if initial_r1 is None:
            r1_init = torch.normal(mean=torch.tensor(0.1), std=torch.tensor(0.02))
        else:
            r1_init = torch.tensor(initial_r1, dtype=torch.float32)
            
        self.raw_r1 = nn.Parameter(r1_init)
        self.raw_r2 = nn.Parameter(torch.tensor(initial_r2, dtype=torch.float32))
        
        parametrize.register_parametrization(self, "raw_r1", NonNegConstraint())
        parametrize.register_parametrization(self, "raw_r2", MinMaxConstraint(0.1, 2.0))

    def forward(self, S):
        r1 = self.raw_r1
        r2 = self.raw_r2
        
        positive_mask = (S > 0).float()
        negative_mask = (S <= 0).float()
        
        return r1 * S * positive_mask + r2 * S * negative_mask

    def regularization_loss(self):
        """Compute L2 regularization loss for the parameters"""
        reg_loss = 0.0
        if self.l2_reg > 0:
            reg_loss += self.l2_reg * torch.sum(self.raw_r1 ** 2)
            reg_loss += self.l2_reg * torch.sum(self.raw_r2 ** 2)
        return reg_loss
    
class MLPMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, num_classes=10, drop_p=0., activation_function='gelu', is_cls_token=False):
        super(MLPMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token

        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1

        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, activation_function) 
            for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_size)

        self.clf = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        out = self.mixer_layers(out)
        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.clf(out)
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, activation_function):
        super(MixerLayer, self).__init__()
        self.activation_mlp1 = nn.GELU() if activation_function == 'gelu' else TReLU()
        self.activation_mlp2 = nn.GELU() if activation_function == 'gelu' else TReLU()
        self.double_activation_mlp1 = nn.GELU() if activation_function == 'gelu' else TReLU()
        self.double_activation_mlp2 = nn.GELU() if activation_function == 'gelu' else TReLU()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, self.activation_mlp1, self.double_activation_mlp1)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, self.activation_mlp2, self.double_activation_mlp2)
    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, activation, double_activation):
        super(MLP1, self).__init__()
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = activation 
        self.double_activation = double_activation
    def forward(self, x):
        out = self.do1(self.act(self.fc1(x)))
        out = self.do2(self.double_activation(self.fc2(out)))
        return out+x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, activation, double_activation):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = activation
        self.double_activation = double_activation
    def forward(self, x):
        out = self.do1(self.act(self.fc1(x)))
        out = self.do2(self.double_activation(self.fc2(out)))
        return out+x
    


if __name__ == '__main__':
    net = MLPMixer(
        in_channels=3,
        img_size=32, 
        patch_size=4, 
        hidden_size=128, 
        hidden_s=512, 
        hidden_c=64, 
        num_layers=8, 
        num_classes=10, 
        drop_p=0.,
        activation_function='gelu',
        is_cls_token=True
        )
    torchsummary.summary(net, (3,32,32))
