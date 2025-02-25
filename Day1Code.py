import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import tifffile

# load image
fn = tifffile.imread('C:/Users/t08154/Downloads/rgb_hsv/AbelDataset/Almonds.tif')
red, green, blue, infra = fn[0, :, :], fn[1, :, :], fn[2, :, :], fn[3, :, :]

# normalise
def normalise(channel):
    return (channel - channel.min()) / (channel.max() - channel.min())

red_norm, green_norm, blue_norm, infra_norm = map(normalise, [red, green, blue, infra])

# stacking rgb
img = np.stack([red_norm, green_norm, blue_norm], axis=-1)
img_uint8 = (img * 255).astype(np.uint8) 

# mask
thresh = threshold_otsu(infra_norm)
mask = (infra_norm < thresh).astype(np.uint8) * 255  

# apply mask
result = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)

# plot before and after
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(img)
ax[0].set_title("Original RGB Image")
ax[0].axis("off")

ax[1].imshow(result)
ax[1].set_title("Masked Image")
ax[1].axis("off")

#convert to hsl and hsv
img_hls = cv2.cvtColor(result, cv2.COLOR_RGB2HLS)
img_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

# Extract HLS and HSV values only for non-masked pixels
mask_nonzero = mask > 0  # Boolean mask for valid pixels

#convert only object pixels
hue_hls = img_hls[:, :, 0][mask_nonzero]
luminance = img_hls[:, :, 1][mask_nonzero]
saturation_hls = img_hls[:, :, 2][mask_nonzero]

hue_hsv = img_hsv[:, :, 0][mask_nonzero]
saturation_hsv = img_hsv[:, :, 1][mask_nonzero]
value = img_hsv[:, :, 2][mask_nonzero]

# flatten channels
hue_hls_flat = hue_hls.flatten()
luminance_flat = luminance.flatten()
saturation_hls_flat = saturation_hls.flatten()

hue_hsv_flat = hue_hsv.flatten()
saturation_hsv_flat = saturation_hsv.flatten()
value_flat = value.flatten()

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# hsl plot
axes[0].scatter(hue_hls_flat, luminance_flat, cmap='hsv', alpha=0.5, marker='.')
axes[0].set_title('HSL: Hue vs Lightness')
axes[0].set_xlabel('Hue')
axes[0].set_ylabel('Lightness')

# hsv plot
axes[1].scatter(hue_hsv_flat, value_flat, cmap='hsv', alpha=0.5, marker='.')
axes[1].set_title('HSV: Hue vs Value')
axes[1].set_xlabel('Hue')
axes[1].set_ylabel('Value')

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()
