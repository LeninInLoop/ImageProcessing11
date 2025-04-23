import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

class ImageUtils:
    @staticmethod
    def load_image(image_path) -> np.ndarray:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        image = Image.open(image_path).convert("L")
        return np.array(image, dtype=np.float32)

    @staticmethod
    def save_image(image: np.ndarray, image_path: str, normalize: bool = False) -> None:
        out = image
        if normalize:
            out = ImageUtils.normalize(out)
        out = np.clip(out, 0, 255).astype(np.uint8)
        Image.fromarray(out).save(image_path)

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        mn, mx = np.min(image), np.max(image)
        if mx > mn:
            return ((image - mn) / (mx - mn) * 255.0).astype(np.float32)
        return np.zeros_like(image, dtype=np.float32)


class DestructionModel:
    @staticmethod
    def get_destruction_model(image: np.ndarray, T: float, a: float, b: float) -> np.ndarray:
        M, N = image.shape

        # Create properly centered frequency coordinates
        u = np.fft.fftfreq(M)[:, np.newaxis] * M  # Scale to match original image
        v = np.fft.fftfreq(N)[np.newaxis, :] * N  # Scale to match original image

        term = u * a + v * b

        H = np.ones((M, N), dtype=complex)

        mask = (term != 0)
        H[mask] = T * np.sin(np.pi * term[mask]) / (np.pi * term[mask]) * np.exp(-1j * np.pi * term[mask])

        # For term=0, H=T
        H[~mask] = T
        return np.fft.fftshift(H)

    @staticmethod
    def apply_destruction(image: np.ndarray, destruction_model: np.ndarray) -> np.ndarray:

        F = np.fft.fft2(image)

        F_shifted = np.fft.fftshift(F)
        G_shifted = F_shifted * destruction_model
        G = np.fft.ifftshift(G_shifted)

        # Inverse FFT to get back to spatial domain
        img_back = np.fft.ifft2(G)
        return np.abs(img_back).astype(np.float32)


def main():
    image_dir = "Images"
    os.makedirs(image_dir, exist_ok=True)
    orig_path = os.path.join(image_dir, "Fig0526(a)(original_DIP)(1).tif")

    # Load
    img = ImageUtils.load_image(orig_path)
    print("Original shape:", img.shape)
    plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis('off'); plt.show()

    # Build model
    T = 1; a = 0.1; b = 0.1
    H = DestructionModel.get_destruction_model(img, T=T, a=a, b=b)
    print("Model shape:", H.shape)

    output_dir = f"Images/T={T},a={a},b={b}"
    os.makedirs(output_dir, exist_ok=True)

    plt.imshow(np.abs(H), cmap="gray"); plt.title("Destruction Model Absolute"); plt.axis('off')
    plt.savefig(f"{output_dir}/Destruction_Model_Absolute.tiff"); plt.show()

    plt.imshow(np.angle(H), cmap="gray"); plt.title("Destruction Model Phase"); plt.axis('off')
    plt.savefig(f"{output_dir}/Destruction_Model_Phase.tiff"); plt.show()

    # Apply
    out = DestructionModel.apply_destruction(img, H)
    plt.imshow(out, cmap="gray"); plt.title(f"After Destruction T={T}, Vx={a}, Vy={b}"); plt.axis('off')
    plt.savefig(f"{output_dir}/Destructed_Image.tiff"); plt.show()

if __name__ == "__main__":
    main()
