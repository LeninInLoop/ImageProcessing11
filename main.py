import os
from typing import Tuple

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

    @staticmethod
    def pad_before_dft(image: np.ndarray) -> np.ndarray:
        new_image_array = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
        new_image_array[0:image.shape[0], 0:image.shape[1]] = image
        return new_image_array

    @staticmethod
    def center_frequency(image: np.ndarray) -> np.ndarray:
        x, y = image.shape
        x, y = np.meshgrid(range(x), range(y))
        mask = np.where( (x + y) % 2 == 0, 1, -1)
        return image * mask

    @staticmethod
    def crop_image(start_idx: Tuple[int, int], stop_idx: Tuple[int, int], image: np.ndarray) -> np.ndarray:
        return image[start_idx[0]:stop_idx[0], start_idx[1]:stop_idx[1]]

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

        F_centered_image = ImageUtils.center_frequency(image)
        F = np.fft.fft2(F_centered_image)

        G = F * destruction_model
        img_back_centered_freq = np.fft.ifft2(G)

        image = ImageUtils.center_frequency(np.real(img_back_centered_freq))

        return image


def main():
    image_dir = "Images"
    os.makedirs(image_dir, exist_ok=True)
    orig_path = os.path.join(image_dir, "Fig0526(a)(original_DIP)(1).tif")

    # Load
    img = ImageUtils.load_image(orig_path)
    print("Original shape:", img.shape)
    plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis('off'); plt.show()

    # Pad original image
    # padded_img = ImageUtils.pad_before_dft(img)
    # print("Padded original image shape", padded_img.shape)
    # plt.imshow(padded_img, cmap="gray"); plt.title("Padded Original Image"); plt.axis('off')
    # plt.savefig(f"Images/Padded_Original_Image.tiff"); plt.show()

    # Build model
    destruction_list = [ # T, a, b
        [1, 0.1, 0.1],
        [1, 0.5, 0.1],
        [1, 0.1, 0.5],
        [1 , 0.5, 0.5],
        [1, 0.1, 0],
        [1, 0, 0.1],
        [5, 0.1, 0.1],
        [5, 0.5, 0.1],
        [5, 0.1, 0.5],
        [5, 0.5, 0.5],
        [5, 0.1, 0],
        [5, 0, 0.1],
        [10, 0.1, 0.1],
        [10, 0.5, 0.1],
        [10, 0.1, 0.5],
        [10, 0.5, 0.5],
        [10, 0.1, 0],
        [10, 0, 0.1],
    ]
    for i in range(len(destruction_list)):
        output_dir = f"Images/T={destruction_list[i][0]},a={destruction_list[i][1]},b={destruction_list[i][2]}"
        os.makedirs(output_dir, exist_ok=True)

        H = DestructionModel.get_destruction_model(
            img,
            T=destruction_list[i][0],
            a=destruction_list[i][1],
            b=destruction_list[i][2]
        )
        print("Model shape:", H.shape) if i == 0 else None

        plt.imshow(np.abs(H), cmap="gray"); plt.title("Destruction Model Absolute"); plt.axis('off')
        plt.savefig(f"{output_dir}/Destruction_Model_Absolute.tiff"); plt.show()

        plt.imshow(np.angle(H), cmap="gray"); plt.title("Destruction Model Phase"); plt.axis('off')
        plt.savefig(f"{output_dir}/Destruction_Model_Phase.tiff"); plt.show()

        # Apply
        out = DestructionModel.apply_destruction(img, H)
        plt.imshow(out, cmap="gray")
        plt.title(
            f"After Destruction T={destruction_list[i][0]},"
            f" Vx={destruction_list[i][1]},"
            f" Vy={destruction_list[i][2]}"
        ); plt.axis('off')
        plt.savefig(f"{output_dir}/Destructed_Image.tiff"); plt.show()

        # Crop
        #out_cropped = ImageUtils.crop_image(start_idx=(0, 0), stop_idx=(img.shape[1], img.shape[0]), image=out)
        #plt.imshow(out_cropped, cmap="gray")
        #plt.title(
        #    f"Cropped-After Destruction T={destruction_list[i][0]},"
        #    f" Vx={destruction_list[i][1]},"
        #    f" Vy={destruction_list[i][2]}"
        #); plt.axis('off')
        #plt.savefig(f"{output_dir}/Destructed_Image_cropped.tiff"); plt.show()

if __name__ == "__main__":
    main()
