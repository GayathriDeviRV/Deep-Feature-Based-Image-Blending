# Packages
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.io import imsave
from torchvision.models import vgg16  # Importing VGG for the second pass

from utils import compute_gt_gradient, make_canvas_mask, numpy2tensor, laplacian_filter_tensor, \
    MeanShift, ResNet50, gram_matrix
import argparse
import os
import imageio.v2 as iio
import torch.nn.functional as F

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--source_file', type=str,
                    default='data/1_source.png', help='path to the source image')
parser.add_argument('--mask_file', type=str,
                    default='data/1_mask.png', help='path to the mask image')
parser.add_argument('--target_file', type=str,
                    default='data/1_target.png', help='path to the target image')
parser.add_argument('--output_dir', type=str,
                    default='results/1', help='path to output')
parser.add_argument('--ss', type=int, default=300, help='source image size')
parser.add_argument('--ts', type=int, default=512, help='target image size')
parser.add_argument('--x', type=int, default=200,
                    help='vertical location (center)')
parser.add_argument('--y', type=int, default=235,
                    help='horizontal location (center)')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_steps', type=int, default=1000,
                    help='Number of iterations in each pass')
parser.add_argument('--save_video', type=bool, default=False,
                    help='save the intermediate reconstruction process')
opt = parser.parse_args()

os.makedirs(opt.output_dir, exist_ok=True)

# Initialize video writer
if opt.save_video:
    recon_process_video = iio.get_writer(os.path.join(
        opt.output_dir, 'output_video.mp4'), format='FFMPEG', mode='I', fps=30)


########### First Pass ###########

# Inputs
source_file = opt.source_file
mask_file = opt.mask_file
target_file = opt.target_file

# Hyperparameter Inputs
gpu_id = opt.gpu_id
num_steps = opt.num_steps
ss = opt.ss  # source image size
ts = opt.ts  # target image size
x_start = opt.x
y_start = opt.y  # blending location

# Default weights for loss functions in the first pass
grad_weight = 1e4
style_weight = 1e4
content_weight = 1
tv_weight = 1e-6

# Load Images
source_img = np.array(Image.open(source_file).convert('RGB').resize((ss, ss)))
target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
mask_img = np.array(Image.open(mask_file).convert('L').resize((ss, ss)))
mask_img[mask_img > 0] = 1

# Make Canvas Mask
canvas_mask = make_canvas_mask(x_start, y_start, target_img, mask_img)
canvas_mask = numpy2tensor(canvas_mask, gpu_id).contiguous()
canvas_mask = canvas_mask.squeeze(0).repeat(
    3, 1, 1).reshape(3, ts, ts).unsqueeze(0)

# Compute Ground-Truth Gradients
gt_gradient = compute_gt_gradient(
    x_start, y_start, source_img, target_img, mask_img, gpu_id)

# Convert Numpy Images Into Tensors
source_img = numpy2tensor(source_img, gpu_id).contiguous()
target_img = numpy2tensor(target_img, gpu_id).contiguous()
input_img = torch.randn(target_img.shape).to(gpu_id)

mask_img = numpy2tensor(mask_img, gpu_id).contiguous()
mask_img = mask_img.squeeze(0).repeat(3, 1, 1).reshape(3, ss, ss).unsqueeze(0)

# Define LBFGS optimizer


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


optimizer = get_input_optimizer(input_img)

# Define Loss Functions
mse = torch.nn.MSELoss()

# Import ResNet50 network for computing style and content loss
mean_shift = MeanShift(gpu_id)
resnet = ResNet50().to(gpu_id)

run = [0]
while run[0] <= num_steps:

    def closure():
        # Composite Foreground and Background to Make Blended Image
        blend_img = input_img * canvas_mask + target_img * (1 - canvas_mask)

        # Compute Laplacian Gradient of Blended Image
        pred_gradient = laplacian_filter_tensor(blend_img, gpu_id)

        # Compute Gradient Loss
        grad_loss = sum(mse(pred_gradient[c], gt_gradient[c])
                        for c in range(len(pred_gradient)))
        grad_loss /= len(pred_gradient)
        grad_loss *= grad_weight

        # Compute Style Loss
        target_features_style = resnet(mean_shift(target_img))
        target_gram_style = [gram_matrix(y) for y in target_features_style]

        blend_features_style = resnet(mean_shift(input_img))
        blend_gram_style = [gram_matrix(y) for y in blend_features_style]

        style_loss = sum(mse(blend_gram_style[layer], target_gram_style[layer]) for layer in range(
            len(blend_gram_style)))
        style_loss /= len(blend_gram_style)
        style_loss *= style_weight

        # Compute Content Loss
        blend_obj = blend_img[:, :, int(
            x_start - ss // 2):int(x_start + ss // 2), int(y_start - ss // 2):int(y_start + ss // 2)]
        source_object_features = resnet(mean_shift(source_img * mask_img))
        blend_object_features = resnet(mean_shift(blend_obj * mask_img))
        content_loss = content_weight * \
            mse(blend_object_features.relu2, source_object_features.relu2)

        # Compute TV Reg Loss
        tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
            torch.sum(
                torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
        tv_loss *= tv_weight

        # Compute Total Loss and Update Image
        loss = grad_loss + style_loss + content_loss + tv_loss
        optimizer.zero_grad()
        loss.backward()

        # Write frames to video
        if opt.save_video:
            current_frame = blend_img[0].clamp(
                0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            recon_process_video.append_data(current_frame)

        # Print Loss
        if run[0] % 10 == 0:
            print(
                f"run {run[0]}: "
                f"grad: {grad_loss.item():.4f}, "
                f"style: {style_loss.item():.4f}, "
                f"content: {content_loss.item():.4f}, "
                f"tv: {tv_loss.item():.4f}"
            )

        run[0] += 1
        return loss

    optimizer.step(closure)

# Save first pass image
blend_img_np = (input_img * canvas_mask + target_img * (1 - canvas_mask)
                )[0].clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
imsave(os.path.join(opt.output_dir, 'first_pass.png'), blend_img_np)

########### Second Pass ###########

# Load VGG model for the second pass


class VGGFeatures(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGGFeatures, self).__init__()
        # Using layers till conv4_3
        vgg_pretrained = vgg16(weights='IMAGENET1K_V1').features[:23]
        self.vgg = vgg_pretrained
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            # Choosing specific layers for style/content representation
            if name in {'4', '9', '16', '23'}:
                out.append(x.contiguous())
        return out


vgg = VGGFeatures().to(gpu_id)
mean_shift = MeanShift(gpu_id)

print("Starting Second Pass Optimization...")

# Set the second pass hyperparameters
style_weight = 1e7
content_weight = 1
tv_weight = 1e-6
num_steps = opt.num_steps

# Load images and set up tensor inputs
first_pass_img_file = os.path.join(opt.output_dir, 'first_pass.png')
first_pass_img = np.array(Image.open(
    first_pass_img_file).convert('RGB').resize((opt.ts, opt.ts)))
target_img = np.array(Image.open(opt.target_file).convert(
    'RGB').resize((opt.ts, opt.ts)))

first_pass_img = torch.from_numpy(first_pass_img).permute(
    2, 0, 1).unsqueeze(0).float().to(gpu_id).contiguous()
target_img = torch.from_numpy(target_img).permute(
    2, 0, 1).unsqueeze(0).float().to(gpu_id).contiguous()

# Initialize optimizer
optimizer = get_input_optimizer(first_pass_img)

# Define loss functions
mse_loss = torch.nn.MSELoss()

run = [0]
while run[0] <= num_steps:

    def closure():
        optimizer.zero_grad()

        # Forward pass through VGG
        target_features = vgg(mean_shift(target_img))
        blend_features = vgg(mean_shift(first_pass_img))

        # Compute Style Loss
        style_loss = 0
        for tf, bf in zip(target_features, blend_features):
            style_loss += mse_loss(gram_matrix(bf), gram_matrix(tf))
        style_loss *= style_weight

        # Compute Content Loss (using only a specific layer for content)
        content_loss = content_weight * \
            mse_loss(blend_features[2], target_features[2])

        # Compute Total Variation (TV) Loss
        tv_loss = tv_weight * (torch.sum(torch.abs(first_pass_img[:, :, :, :-1] - first_pass_img[:, :, :, 1:])) +
                               torch.sum(torch.abs(first_pass_img[:, :, :-1, :] - first_pass_img[:, :, 1:, :])))

        # Combine losses
        loss = style_loss + content_loss + tv_loss
        loss.backward()

        # Write frames to video
        if opt.save_video:
            current_frame = first_pass_img[0].clamp(
                0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            recon_process_video.append_data(current_frame)

        # Print Loss
        if run[0] % 50 == 0:
            print(
                f"Second pass run {run[0]}: "
                f"style: {style_loss.item():.4f}, "
                f"content: {content_loss.item():.4f}, "
                f"tv: {tv_loss.item():.4f}"
            )

        run[0] += 1
        return loss

    optimizer.step(closure)

# Save the final output
first_pass_img.data.clamp_(0, 255)
output_img_np = first_pass_img.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
output_img_np = np.clip(output_img_np, 0, 255).astype(np.uint8)
output_img_path = os.path.join(opt.output_dir, 'second_pass.png')
imsave(output_img_path, output_img_np)

# Close video writer if enabled
if opt.save_video:
    recon_process_video.close()
    print("Reconstruction video saved.")

print("Second pass completed and saved.")
