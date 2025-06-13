# Understanding the Texture Analysis Methodology

This methodology divides texture analysis into two main steps: **Local Segmentation** and **Global Clustering**. Let's explore each one and how they are implemented in the code.

---

## 1. Local Segmentation: Identifying Textures within Each Image

Local segmentation aims to divide each image into smaller regions (blocks) and identify different types of texture within these regions.

### Methodology Steps

1. **Grayscale Conversion:** The original image is converted to grayscale. This simplifies the analysis, focusing on the texture features rather than the color.

2. **Filter Bank Application:** A set of filters is applied to the grayscale image to enhance various texture features.

- **12 Gabor Filters:** Used to detect edges and patterns in different orientations and scales. There are 4 orientations (0°, 45°, 90°, 135°) and 3 scales, totaling $4 \times 3 = 12$ Gabor filters.
- **3 Laplacian Gaussian Filters (LoG):** These are circular filters sensitive to points and lines, applied at 3 different scales.

3. **Division into Blocks and Extraction of Local Features:**

- The image is divided into non-overlapping blocks of $16 \times 16$ pixels.
- For each block, the **average response of each of the 15 filters** (12 Gabor + 3 LoG) is calculated. This generates a **15-dimensional vector** for each block, which represents the predominant texture of that small region.

4. **K-Means Clustering (Segmentation):**

- The feature vectors of all blocks in the image are clustered using the **K-Means** algorithm, with $k=4$.
- This assigns each block one of four texture labels, creating a segmentation map where blocks with similar textures receive the same label.

### Code Implementation

1. **Filter Generation:**

```python
def build_filter_bank(
    scales=[4, 8, 16], orientations=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    ):
    # ... (code to build the Gabor and LoG kernels) ...
    return filters
```

- The `filters` variable is a **list containing 15 2D NumPy arrays**. Each matrix represents a filter kernel (Gabor or LoG) ready to be applied to the image.

2. **Applying Filters:**

```python
def compute_filter_responses(gray_img, filters):
    # ... (code to apply each filter from the 'filters' list to the image) ...
    return responses
```

- The `responses` variable is a **list of 15 response images**. Each response image is a 2D NumPy array of the same size as the original image, where each pixel indicates the magnitude of the response of a specific filter at that location in the image.

3. **Calculation of Local Descriptors and Segmentation:**

```python
def segment_texture(responses, num_segments=4, window_size=16):
    # ... (code to divide into blocks, calculate averages and apply K-Means) ...
    return seg_map, kmeans
```

- Inside this function, the `responses` are used to calculate a **vector of 15 features for each block** ($16 \times 16$ pixels). These vectors are grouped by K-Means, generating the `seg_map` (local segmentation map of the texture).

---

## 2. Global Clustering: Categorizing Images by Overall Texture

Global clustering aims to categorize entire images based on their global textural features, creating groups of scenes with similar textures.

### Methodology Steps

1. **Image-Level Descriptor Extraction:** For each image, a “global descriptor” is computed. This is done by taking the **average of the response of each of the 15 filters over the _entire scene_**. This results in a single vector of 15 components per image, which summarizes the textural characteristics of the entire scene.

2. **K-Means Clustering (Image Clustering):** The global descriptors of **all 20 images** are then subjected to K-Means again, with $k=4$. This groups the images into four categories, where each category contains images with similar overall textural characteristics.

### Implementation in Code

1. **Calculating Global Descriptors:**

```python
def extract_global_descriptor(responses):
    desc = np.array([resp.mean() for resp in responses])
    return desc
```

- The `desc` variable is a **1D NumPy array of 15 positions**. Each position stores the mean of the response of one of the 15 filters, calculated over _the entire image_.

2. **Image Clustering (in `main` function):**

```python
def main():
    # ... (loop over all images) ...
    descriptors = [] # List that stores the global descriptors of each image
    # ...
    # For each image, 'desc' is generated and added to 'descriptors'
    descriptors.append(desc)
    # ...
    X = np.vstack(descriptors) # Convert the list into a single matrix for K-Means
    img_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = img_kmeans.labels_
    # ... (code to organize images into cluster folders) ...
```

- The `descriptors` variable is a **list of 1D NumPy arrays**. Each 1D array is the global descriptor of 15 components of an image. If you have 20 images, this list will have 20 of these arrays.
- `np.vstack(descriptors)` creates a 2D matrix where each row is the global descriptor of an image. This matrix is ​​then used by K-Means to group images into 4 categories, based on their overall textures.

---

This robust pipeline allows you to analyze and organize images both at the level of texture regions and their overall similarity.
