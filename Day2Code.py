import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import tifffile
from sklearn.cluster import KMeans

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

img_hls = cv2.cvtColor(result, cv2.COLOR_RGB2HLS)
img_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

# Extract HLS and HSV values only for non-masked pixels
mask_nonzero = mask > 0  # Boolean mask for valid pixels

hue_hls = img_hls[:, :, 0][mask_nonzero]
luminance = img_hls[:, :, 1][mask_nonzero]
saturation_hls = img_hls[:, :, 2][mask_nonzero]

hue_hsv = img_hsv[:, :, 0][mask_nonzero]
saturation_hsv = img_hsv[:, :, 1][mask_nonzero]
value = img_hsv[:, :, 2][mask_nonzero]

# Flatten channels
hue_hls_flat = hue_hls.flatten()
luminance_flat = luminance.flatten()
saturation_hls_flat = saturation_hls.flatten()

hue_hsv_flat = hue_hsv.flatten()
saturation_hsv_flat = saturation_hsv.flatten()
value_flat = value.flatten()

# Normalize hue for HSL and HSV
hue_hls_flat = [val + 90 if val < 90 else val - 90 for val in hue_hls_flat]
hue_hsv_flat = [val + 90 if val < 90 else val - 90 for val in hue_hsv_flat]

hue_hls_norm = np.array(hue_hls_flat) / 179.0
hue_hsv_norm = np.array(hue_hsv_flat) / 179.0

data = np.column_stack((hue_hls_norm, luminance_flat, hue_hsv_norm, value_flat))

# Apply KMeans clustering to identify outliers
kmeans = KMeans(n_clusters=257, random_state=0, n_init=20).fit(data)
labels = kmeans.labels_

# Compute distances from each point to its assigned cluster center
distances = kmeans.transform(data).min(axis=1)

# Define an outlier threshold (e.g., farthest 5% of points)
outlier_threshold = np.percentile(distances, 87)  
outlier_mask = distances > outlier_threshold

# Remove outliers
clean_data = data[~outlier_mask]

# Create plots after outlier removal
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# HSL: Hue vs Lightness (after removing outliers)
axes[0].scatter(clean_data[:, 0], clean_data[:, 1], c=clean_data[:, 0], cmap='hsv', alpha=0.5, marker='.')
axes[0].set_title('HSL: Hue vs Lightness')
axes[0].set_xlabel('Hue')
axes[0].set_ylabel('Lightness')

# HSV: Hue vs Value (after removing outliers)
axes[1].scatter(clean_data[:, 2], clean_data[:, 3], c=clean_data[:, 2], cmap='hsv', alpha=0.5, marker='.')
axes[1].set_title('HSV: Hue vs Value')
axes[1].set_xlabel('Hue')
axes[1].set_ylabel('Value')

plt.tight_layout()
plt.show()