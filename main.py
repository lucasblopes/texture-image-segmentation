import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from pathlib import Path
import glob


class TextureAnalyzer:
    def __init__(self, scales=[3, 7, 15], window_size=16, n_clusters=5):
        """
        Initialize the texture analyzer with specified parameters.

        Args:
            scales: List of kernel sizes for filters
            window_size: Size of non-overlapping square windows
            n_clusters: Number of clusters for K-Means
        """
        self.scales = scales
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.filters = self._create_filters()

    def _create_filters(self):
        """Create texture filters at different orientations and scales."""
        filters = []

        for scale in self.scales:
            # Horizontal filter (0 degrees)
            horizontal = np.zeros((scale, scale))
            horizontal[scale // 2, :] = 1
            filters.append(("horizontal_" + str(scale), horizontal))

            # Vertical filter (90 degrees)
            vertical = np.zeros((scale, scale))
            vertical[:, scale // 2] = 1
            filters.append(("vertical_" + str(scale), vertical))

            # Diagonal filter (45 degrees)
            diagonal45 = np.zeros((scale, scale))
            for i in range(scale):
                if i < scale:
                    diagonal45[i, i] = 1
            filters.append(("diagonal45_" + str(scale), diagonal45))

            # Diagonal filter (135 degrees)
            diagonal135 = np.zeros((scale, scale))
            for i in range(scale):
                if i < scale:
                    diagonal135[i, scale - i - 1] = 1
            filters.append(("diagonal135_" + str(scale), diagonal135))

            # Circular/isotropic filter - FIXED VERSION
            circular = np.zeros((scale, scale))
            center = scale // 2
            radius = scale // 4

            # Create circle using distance from center
            for i in range(scale):
                for j in range(scale):
                    # Calculate distance from center
                    if ((i - center) ** 2 + (j - center) ** 2) <= radius**2:
                        circular[i, j] = 1

            filters.append(("circular_" + str(scale), circular))

        return filters

    def apply_filters(self, image):
        """
        Apply all filters to the input image.

        Args:
            image: Grayscale input image

        Returns:
            Dictionary of filter responses
        """
        responses = {}

        for name, kernel in self.filters:
            # Normalize kernel for consistent filtering
            kernel = kernel / np.sum(kernel) if np.sum(kernel) != 0 else kernel
            # Apply filter using convolution
            response = cv2.filter2D(image, -1, kernel)
            responses[name] = response

        return responses

    def extract_features(self, responses):
        """
        Extract features from filter responses using non-overlapping windows.

        Args:
            responses: Dictionary of filter responses

        Returns:
            Feature vectors and their corresponding positions
        """
        # Get dimensions from first response
        first_key = list(responses.keys())[0]
        height, width = responses[first_key].shape

        # Calculate number of windows
        n_windows_h = height // self.window_size
        n_windows_w = width // self.window_size

        # Initialize feature array
        n_filters = len(responses)
        features = np.zeros((n_windows_h * n_windows_w, n_filters))
        positions = np.zeros((n_windows_h * n_windows_w, 2), dtype=int)

        # Extract features from each window
        for i in range(n_windows_h):
            for j in range(n_windows_w):
                window_idx = i * n_windows_w + j
                positions[window_idx] = [i, j]

                filter_idx = 0
                for name, response in responses.items():
                    # Extract window from response
                    window = response[
                        i * self.window_size : (i + 1) * self.window_size,
                        j * self.window_size : (j + 1) * self.window_size,
                    ]
                    # Calculate mean response
                    features[window_idx, filter_idx] = np.mean(window)
                    filter_idx += 1

        return features, positions

    def cluster_features(self, features):
        """
        Cluster feature vectors using K-Means.

        Args:
            features: Array of feature vectors

        Returns:
            Cluster assignments
        """
        # Normalize features for better clustering
        features_norm = (features - features.mean(axis=0)) / features.std(axis=0)
        features_norm = np.nan_to_num(features_norm)  # Handle potential NaN values

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_norm)

        return clusters

    def visualize_clusters(self, clusters, positions, image_shape):
        """
        Create visualization of cluster assignments.

        Args:
            clusters: Cluster assignments
            positions: Window positions
            image_shape: Shape of original image

        Returns:
            Visualization image
        """
        # Create empty visualization image
        height, width = image_shape
        n_windows_h = height // self.window_size
        n_windows_w = width // self.window_size

        # Create colormap for visualization
        cluster_vis = np.zeros((n_windows_h, n_windows_w), dtype=np.uint8)

        # Assign cluster IDs to windows
        for idx, (i, j) in enumerate(positions):
            if i < n_windows_h and j < n_windows_w:
                cluster_vis[i, j] = clusters[idx]

        # Scale to full image size
        cluster_vis_full = np.zeros(
            (n_windows_h * self.window_size, n_windows_w * self.window_size),
            dtype=np.uint8,
        )

        for i in range(n_windows_h):
            for j in range(n_windows_w):
                cluster_vis_full[
                    i * self.window_size : (i + 1) * self.window_size,
                    j * self.window_size : (j + 1) * self.window_size,
                ] = cluster_vis[i, j]

        # Scale cluster IDs to 0-255 for better visualization
        cluster_vis_full = (cluster_vis_full * (255 // (self.n_clusters - 1))).astype(
            np.uint8
        )

        return cluster_vis_full

    def process_image(self, image):
        """
        Process a single image through the complete pipeline.

        Args:
            image: Input grayscale image

        Returns:
            Original image, filter responses, and cluster visualization
        """
        # Apply filters
        responses = self.apply_filters(image)

        # Extract features
        features, positions = self.extract_features(responses)

        # Cluster features
        clusters = self.cluster_features(features)

        # Create visualization
        cluster_vis = self.visualize_clusters(clusters, positions, image.shape)

        return image, responses, cluster_vis

    def process_and_visualize(self, image_path, save_dir=None):
        """
        Process an image and visualize the results.

        Args:
            image_path: Path to input image
            save_dir: Directory to save results (optional)
        """
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Failed to load image: {image_path}")
            return

        # Make sure image dimensions are multiples of window_size
        h, w = img.shape
        new_h = (h // self.window_size) * self.window_size
        new_w = (w // self.window_size) * self.window_size
        img = img[:new_h, :new_w]

        # Process image
        orig_img, responses, cluster_vis = self.process_image(img)

        # Create visualization
        plt.figure(figsize=(20, 15))

        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(orig_img, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        # Sample filter responses (showing one from each scale)
        shown_scales = set()
        resp_idx = 2
        for name, response in responses.items():
            scale = name.split("_")[1]
            if scale not in shown_scales and resp_idx <= 5:
                shown_scales.add(scale)
                plt.subplot(2, 3, resp_idx)
                plt.imshow(response, cmap="jet")
                plt.title(f"Filter: {name}")
                plt.axis("off")
                resp_idx += 1

        # Cluster visualization
        plt.subplot(2, 3, 6)
        plt.imshow(cluster_vis, cmap="viridis")
        plt.title("Texture Clusters")
        plt.axis("off")

        # Save or show results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.basename(image_path).split(".")[0]
            plt.savefig(os.path.join(save_dir, f"{base_name}_results.png"))

            # Also save cluster visualization separately
            cv2.imwrite(
                os.path.join(save_dir, f"{base_name}_clusters.png"), cluster_vis
            )
        else:
            plt.tight_layout()
            plt.show()

        plt.close()

    def batch_process(self, image_dir, output_dir):
        """
        Process a batch of images.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Find all image files
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))

        print(f"Found {len(image_paths)} images to process")

        # Process each image
        for i, img_path in enumerate(image_paths):
            print(
                f"Processing image {i + 1}/{len(image_paths)}: {os.path.basename(img_path)}"
            )
            self.process_and_visualize(img_path, output_dir)


# Usage example
if __name__ == "__main__":
    # Create texture analyzer
    analyzer = TextureAnalyzer(
        scales=[3, 9, 15],  # Small, medium, large scales
        window_size=16,  # 16x16 windows
        n_clusters=7,  # Number of texture clusters
    )

    # Example directories
    input_dir = "images"
    output_dir = "results"

    # Process all images in directory
    analyzer.batch_process(input_dir, output_dir)

