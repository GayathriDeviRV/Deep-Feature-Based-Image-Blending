# Neural Image Blending with Gradient and Style Loss

This project implements a two-pass neural image blending algorithm that combines gradient-based optimization and neural style transfer techniques. The code is designed to seamlessly blend a source object into a target background using a provided mask, while maintaining natural-looking boundaries and textures.

---

## **Key Features**

1. **Custom Gradient Loss**:
   - Utilizes Laplacian filtering to compute gradients for blending.
   - Ensures smooth gradient transitions between source and target regions.

2. **Two-Pass Optimization**:
   - **First Pass:** Focuses on blending gradients of the source and target.
   - **Second Pass:** Refines the output with neural style transfer using VGG features.

3. **Pre-trained Networks**:
   - **ResNet50** for feature extraction in the first pass.
   - **VGG16** for style and content representation in the second pass.

4. **Loss Functions**:
   - **Gradient Loss:** Enforces consistency with ground-truth gradients.
   - **Style Loss:** Matches textures using Gram matrices.
   - **Content Loss:** Retains source object details.
   - **Total Variation Loss:** Promotes smoothness in the output.

5. **GPU Acceleration**:
   - The implementation leverages PyTorch tensors to run computations on a GPU.

6. **Optional Video Output**:
   - Saves the blending process as a video for visualization.

---

## **Installation**

### **Requirements**
- Python 3.7+
- PyTorch
- torchvision
- numpy
- scikit-image
- PIL (Pillow)
- imageio

---

## How It Works

### Inputs
- **Source Image**: The object to be blended into the target.
- **Mask Image**: A binary mask highlighting the source object.
- **Target Image**: The background where the source will be blended.

### Outputs
- **First Pass Image**: Blends gradients from the source and target.
- **Second Pass Image**: Refines textures and harmonizes the blend using neural style transfer.

### Pipeline Overview
#### First Pass:
- Combines source and target images using gradient consistency.
- Computes losses: Gradient Loss, Style Loss, Content Loss, and TV Loss.

#### Second Pass:
- Further refines textures and harmonization using VGG-based style transfer.

---

## Usage

### Running the Code
To run the blending pipeline, execute the following command:
```bash
!python run.py \
    --source_file data/ballons_source.png \
    --mask_file data/ballons_mask.png \
    --target_file data/green_target.png \
    --output_dir results/res_1 \
    --ss 240 \
    --ts 512 \
    --x 200 \
    --y 250 \
    --gpu_id 0 \
    --num_steps 1100 \
    --save_video True
```

### Command-line Arguments

- `--source_file`: Path to the source image.
- `--mask_file`: Path to the mask image.
- `--target_file`: Path to the target image.
- `--output_dir`: Directory to save the outputs.
- `--ss`: Size of the source image (rescaled width/height).
- `--ts`: Size of the target image (rescaled width/height).
- `--x`: Vertical location of the blending center.
- `--y`: Horizontal location of the blending center.
- `--gpu_id`: GPU ID to use for acceleration.
- `--num_steps`: Number of optimization iterations per pass.
- `--save_video`: Whether to save a video of the blending process (`True` or `False`).

---

## **Output**

