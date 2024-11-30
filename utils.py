import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple


def numpy2tensor(np_array, gpu_id):
    """Convert numpy array to tensor and move it to the specified GPU."""
    if len(np_array.shape) == 2:
        tensor = torch.from_numpy(np_array).unsqueeze(0).float().to(gpu_id)
    else:
        tensor = torch.from_numpy(np_array).unsqueeze(
            0).contiguous().transpose(1, 3).transpose(2, 3).float().to(gpu_id)
    return tensor


def make_canvas_mask(x_start, y_start, target_img, mask):
    """Create a canvas mask for blending."""
    canvas_mask = np.zeros((target_img.shape[0], target_img.shape[1]))
    canvas_mask[int(x_start - mask.shape[0] * 0.5):int(x_start + mask.shape[0] * 0.5),
                int(y_start - mask.shape[1] * 0.5):int(y_start + mask.shape[1] * 0.5)] = mask
    return canvas_mask


def laplacian_filter_tensor(img_tensor, gpu_id):
    """Apply a Laplacian filter to the image tensor."""
    laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_conv = nn.Conv2d(
        1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    laplacian_conv.weight = nn.Parameter(torch.from_numpy(
        laplacian_filter).float().unsqueeze(0).unsqueeze(0).to(gpu_id))

    for param in laplacian_conv.parameters():
        param.requires_grad = False

    red_img_tensor = img_tensor[:, 0, :, :].unsqueeze(1)
    green_img_tensor = img_tensor[:, 1, :, :].unsqueeze(1)
    blue_img_tensor = img_tensor[:, 2, :, :].unsqueeze(1)

    red_gradient_tensor = laplacian_conv(red_img_tensor).squeeze(1)
    green_gradient_tensor = laplacian_conv(green_img_tensor).squeeze(1)
    blue_gradient_tensor = laplacian_conv(blue_img_tensor).squeeze(1)
    return red_gradient_tensor, green_gradient_tensor, blue_gradient_tensor


def compute_gt_gradient(x_start, y_start, source_img, target_img, mask, gpu_id):
    """Compute the ground truth gradient for the blended image."""
    # Convert images to tensors
    source_img_tensor = numpy2tensor(source_img, gpu_id)
    target_img_tensor = numpy2tensor(target_img, gpu_id)

    # Apply Laplacian filter
    red_source_gradient_tensor, green_source_gradient_tensor, blue_source_gradient_tensor = laplacian_filter_tensor(
        source_img_tensor, gpu_id)
    red_target_gradient_tensor, green_target_gradient_tensor, blue_target_gradient_tensor = laplacian_filter_tensor(
        target_img_tensor, gpu_id)

    # Convert to numpy for further processing
    red_source_gradient = red_source_gradient_tensor.cpu().data.numpy()[0]
    green_source_gradient = green_source_gradient_tensor.cpu().data.numpy()[0]
    blue_source_gradient = blue_source_gradient_tensor.cpu().data.numpy()[0]
    red_target_gradient = red_target_gradient_tensor.cpu().data.numpy()[0]
    green_target_gradient = green_target_gradient_tensor.cpu().data.numpy()[0]
    blue_target_gradient = blue_target_gradient_tensor.cpu().data.numpy()[0]

    # Create a canvas mask
    canvas_mask = make_canvas_mask(x_start, y_start, target_img, mask)

    # Blend gradients
    def apply_mask_and_blend(color_gradient, color_target_gradient):
        color_gradient_fg = np.zeros(canvas_mask.shape)
        color_gradient_fg[int(x_start - source_img.shape[0] * 0.5):int(x_start + source_img.shape[0] * 0.5),
                          int(y_start - source_img.shape[1] * 0.5):int(y_start + source_img.shape[1] * 0.5)] = color_gradient * mask
        color_gradient_bg = color_target_gradient * (canvas_mask - 1) * (-1)
        return numpy2tensor(color_gradient_fg + color_gradient_bg, gpu_id)

    gt_red_gradient = apply_mask_and_blend(
        red_source_gradient, red_target_gradient)
    gt_green_gradient = apply_mask_and_blend(
        green_source_gradient, green_target_gradient)
    gt_blue_gradient = apply_mask_and_blend(
        blue_source_gradient, blue_target_gradient)

    return [gt_red_gradient, gt_green_gradient, gt_blue_gradient]


class ResNet50(torch.nn.Module):
    """ResNet50 model for feature extraction."""

    def __init__(self, requires_grad=False):
        super(ResNet50, self).__init__()
        resnet_pretrained = models.resnet50(pretrained=True)
        self.layer1 = torch.nn.Sequential(
            *list(resnet_pretrained.children())[:4])  # Conv1
        self.layer2 = torch.nn.Sequential(
            *list(resnet_pretrained.children())[4])   # Layer1
        self.layer3 = torch.nn.Sequential(
            *list(resnet_pretrained.children())[5])   # Layer2
        self.layer4 = torch.nn.Sequential(
            *list(resnet_pretrained.children())[6])   # Layer3
        self.layer5 = torch.nn.Sequential(
            *list(resnet_pretrained.children())[7])   # Layer4

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.layer1(X)
        h_relu1 = h
        h = self.layer2(h)
        h_relu2 = h
        h = self.layer3(h)
        h_relu3 = h
        h = self.layer4(h)
        h_relu4 = h
        h = self.layer5(h)
        h_relu5 = h

        ResNetOutputs = namedtuple(
            "ResNetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        return ResNetOutputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


def gram_matrix(y):
    """Compute Gram matrix for style loss."""
    (b, ch, h, w) = y.size()
    features = y.reshape(b, ch, w * h)
    features_t = features.transpose(1, 2)
    return features.bmm(features_t) / (ch * h * w)


class MeanShift(nn.Conv2d):
    """Mean shift layer for normalizing image features."""

    def __init__(self, gpu_id):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        rgb_range = 1
        rgb_mean = (0.485, 0.456, 0.406)
        rgb_std = (0.229, 0.224, 0.225)
        sign = -1
        std = torch.Tensor(rgb_std).to(gpu_id)
        self.weight.data = torch.eye(3).reshape(
            3, 3, 1, 1).to(gpu_id) / std.reshape(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * \
            torch.Tensor(rgb_mean).to(gpu_id) / std
        for p in self.parameters():
            p.requires_grad = False


def get_matched_features_numpy(blended_features, target_features):
    matched_features = blended_features.new_full(
        size=blended_features.size(), fill_value=0, requires_grad=False)
    cpu_blended_features = blended_features.cpu().detach().numpy()
    cpu_target_features = target_features.cpu().detach().numpy()
    for filter in range(0, blended_features.size(1)):
        matched_filter = torch.from_numpy(hist_match_numpy(cpu_blended_features[0, filter, :, :],
                                                           cpu_target_features[0, filter, :, :])).to(blended_features.device)
        matched_features[0, filter, :, :] = matched_filter
    return matched_features


def get_matched_features_pytorch(blended_features, target_features):
    matched_features = blended_features.new_full(size=blended_features.size(
    ), fill_value=0, requires_grad=False).to(blended_features.device)
    for filter in range(0, blended_features.size(1)):
        matched_filter = hist_match_pytorch(
            blended_features[0, filter, :, :], target_features[0, filter, :, :])
        matched_features[0, filter, :, :] = matched_filter
    return matched_features


def hist_match_pytorch(source, template):

    oldshape = source.size()
    source = source.contiguous().reshape(-1)
    template = template.contiguous().reshape(-1)

    max_val = max(source.max().item(), template.max().item())
    min_val = min(source.min().item(), template.min().item())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        return source.contiguous().reshape(oldshape)

    hist_bin_centers = torch.arange(
        start=min_val, end=max_val, step=hist_step).to(source.device)
    hist_bin_centers = hist_bin_centers + hist_step / 2.0

    source_hist = torch.histc(
        input=source, min=min_val, max=max_val, bins=num_bins)
    template_hist = torch.histc(
        input=template, min=min_val, max=max_val, bins=num_bins)

    source_quantiles = torch.cumsum(input=source_hist, dim=0)
    source_quantiles = source_quantiles / source_quantiles[-1]

    template_quantiles = torch.cumsum(input=template_hist, dim=0)
    template_quantiles = template_quantiles / template_quantiles[-1]

    nearest_indices = torch.argmin(torch.abs(template_quantiles.repeat(len(
        source_quantiles), 1) - source_quantiles.contiguous().reshape(-1, 1).repeat(1, len(template_quantiles))), dim=1)

    source_bin_index = torch.clamp(input=torch.round(
        source / hist_step), min=0, max=num_bins - 1).long()

    mapped_indices = torch.gather(
        input=nearest_indices, dim=0, index=source_bin_index)
    matched_source = torch.gather(
        input=hist_bin_centers, dim=0, index=mapped_indices)

    return matched_source.contiguous().reshape(oldshape)


async def hist_match_pytorch_async(source, template, index, storage):

    oldshape = source.size()
    source = source.reshape(-1)
    template = template.reshape(-1)

    max_val = max(source.max().item(), template.max().item())
    min_val = min(source.min().item(), template.min().item())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        storage[0, index, :, :] = source.reshape(oldshape)
        return

    hist_bin_centers = torch.arange(
        start=min_val, end=max_val, step=hist_step).to(source.device)
    hist_bin_centers = hist_bin_centers + hist_step / 2.0

    source_hist = torch.histc(
        input=source, min=min_val, max=max_val, bins=num_bins)
    template_hist = torch.histc(
        input=template, min=min_val, max=max_val, bins=num_bins)

    source_quantiles = torch.cumsum(input=source_hist, dim=0)
    source_quantiles = source_quantiles / source_quantiles[-1]

    template_quantiles = torch.cumsum(input=template_hist, dim=0)
    template_quantiles = template_quantiles / template_quantiles[-1]

    nearest_indices = torch.argmin(torch.abs(template_quantiles.repeat(len(
        source_quantiles), 1) - source_quantiles.contiguous().reshape(-1, 1).repeat(1, len(template_quantiles))), dim=1)

    source_bin_index = torch.clamp(input=torch.round(
        source / hist_step), min=0, max=num_bins - 1).long()

    mapped_indices = torch.gather(
        input=nearest_indices, dim=0, index=source_bin_index)
    matched_source = torch.gather(
        input=hist_bin_centers, dim=0, index=mapped_indices)

    storage[0, index, :, :] = matched_source.contiguous().reshape(oldshape)


async def loop_features_pytorch(source, target, storage):
    size = source.shape
    tasks = []

    for i in range(0, size[1]):
        task = asyncio.ensure_future(hist_match_pytorch_async(
            source[0, i], target[0, i], i, storage))
        tasks.append(task)

    await asyncio.gather(*tasks)


def get_matched_features_pytorch_async(source, target, matched):
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(
        loop_features_pytorch(source, target, matched))
    loop.run_until_complete(future)
    loop.close()


def hist_match_numpy(source, template):

    oldshape = source.shape

    source = source.ravel()
    template = template.ravel()

    max_val = max(source.max(), template.max())
    min_val = min(source.min(), template.min())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        return source.contiguous().reshape(oldshape)

    source_hist, source_bin_edges = np.histogram(
        a=source, bins=num_bins, range=(min_val, max_val))
    template_hist, template_bin_edges = np.histogram(
        a=template, bins=num_bins, range=(min_val, max_val))

    hist_bin_centers = source_bin_edges[:-1] + hist_step / 2.0

    source_quantiles = np.cumsum(source_hist).astype(np.float32)
    source_quantiles /= source_quantiles[-1]
    template_quantiles = np.cumsum(template_hist).astype(np.float32)
    template_quantiles /= template_quantiles[-1]

    index_function = np.vectorize(
        pyfunc=lambda x: np.argmin(np.abs(template_quantiles - x)))

    nearest_indices = index_function(source_quantiles)

    source_data_bin_index = np.clip(a=np.round(
        source / hist_step), a_min=0, a_max=num_bins-1).astype(np.int32)

    mapped_indices = np.take(nearest_indices, source_data_bin_index)
    matched_source = np.take(hist_bin_centers, mapped_indices)

    return matched_source.contiguous().reshape(oldshape)


def main():
    size = (64, 512, 512)
    source = np.random.randint(
        low=0, high=500000, size=size).astype(np.float32)
    target = np.random.randint(
        low=0, high=500000, size=size).astype(np.float32)
    source_tensor = torch.Tensor(source).to(0)
    target_tensor = torch.Tensor(target).to(0)
    matched_numpy = np.zeros(shape=size)
    matched_pytorch = torch.zeros(size=size, device=0)

    numpy_time = time.process_time()

    for i in range(0, size[0]):
        matched_numpy[i, :, :] = hist_match_numpy(source[i], target[i])

    numpy_time = time.process_time() - numpy_time

    pytorch_time = time.process_time()

    for i in range(0, size[0]):
        matched_pytorch[i, :, :] = hist_match_pytorch(
            source_tensor[i], target_tensor[i])

    pytorch_time = time.process_time() - pytorch_time


if __name__ == "__main__":
    main()
