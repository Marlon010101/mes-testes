from PIL import Image, ImageFilter
import numpy as np

source = Image.open("assets/logo-new-source.jpeg").convert("RGB")
pixels = np.asarray(source, dtype=np.float32)
background = np.asarray(source.filter(ImageFilter.GaussianBlur(28)), dtype=np.float32)
delta = pixels - background
colors = [np.array([255, 255, 255], dtype=np.float32), np.array([255, 190, 0], dtype=np.float32)]
alphas, errors = [], []
for foreground in colors:
    direction = foreground - background
    denominator = np.maximum(np.sum(direction * direction, axis=2), 1)
    alpha = np.clip(np.sum(delta * direction, axis=2) / denominator, 0, 1)
    alphas.append(alpha)
    errors.append(np.sum((delta - alpha[:, :, None] * direction) ** 2, axis=2))

use_yellow = errors[1] < errors[0]
alpha = np.where(use_yellow, alphas[1], alphas[0])
alpha = np.where(alpha < 0.035, 0, alpha)
foreground = np.where(use_yellow[:, :, None], colors[1], colors[0]).astype(np.uint8)
rgba = np.dstack((foreground, (alpha * 255).astype(np.uint8)))
image = Image.fromarray(rgba, "RGBA")
logo = image.crop(image.getbbox())
logo.save("assets/logo-climed.png", optimize=True)

crop = image.crop((690, 285, 1095, 690))
mark = crop.crop(crop.getbbox())
side = max(mark.size)
square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
square.alpha_composite(mark, ((side - mark.width) // 2, (side - mark.height) // 2))
square.resize((256, 256), Image.Resampling.LANCZOS).save("assets/logo-mark.png", optimize=True)
print("Processed logo:", logo.size)
