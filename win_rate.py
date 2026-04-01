from pathlib import Path
from PIL import Image
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Utility — Tight Crop Context
# ============================================================

def get_tight_crop(img: Image.Image):
    """
    Returns (cropped_img, bbox) where bbox = (x1, y1, x2, y2).
    bbox is the bounding box of non-transparent pixels.
    """
    alpha = np.array(img)[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any() or not cols.any():
        # Fully transparent layer
        return img, (0, 0, img.width, img.height)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    cropped = img.crop((x1, y1, x2 + 1, y2 + 1))
    return cropped, (int(x1), int(y1), int(x2) + 1, int(y2) + 1)


class LayerCropContext:
    """
    Crops to the tight bounding box of the object,
    lets you operate on just the object region,
    then restores back to full canvas size.

    Usage:
        ctx = LayerCropContext(img)
        obj = ctx.get_object()
        obj = some_operation(obj)
        img = ctx.restore(obj)
    """

    def __init__(self, img: Image.Image):
        self.original_size = img.size
        self.cropped, self.bbox = get_tight_crop(img)

    def get_object(self) -> Image.Image:
        return self.cropped

    def get_mask(self) -> np.ndarray:
        """Returns alpha mask of the cropped object."""
        return np.array(self.cropped)[:, :, 3]

    def restore(self, modified_crop: Image.Image) -> Image.Image:
        """Pastes the modified object back onto a blank full-canvas image."""
        canvas = Image.new('RGBA', self.original_size, (0, 0, 0, 0))
        x1, y1 = self.bbox[0], self.bbox[1]
        canvas.paste(modified_crop, (x1, y1), modified_crop)
        return canvas


# ============================================================
# Layer Renderer
# ============================================================

class LayerRenderer:
    def __init__(self, dataset):
        self.dataset = dataset

    # --------------------------------------------------------
    # Individual property appliers
    # Each works on full canvas RGBA image unless noted
    # --------------------------------------------------------

    def _apply_geometric(self, img: Image.Image, instance: dict) -> Image.Image:
        """Flip, scale, rotation, shear, perspective, elastic, crop."""

        # --- Flip ---
        if instance.get('flip_horizontal'):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if instance.get('flip_vertical'):
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # --- Scale ---
        scale = instance.get('scale', 1.0)
        if scale != 1.0:
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # --- Rotation ---
        rotation = instance.get('rotation', 0)
        if rotation != 0:
            img = img.rotate(
                rotation,
                resample=Image.BICUBIC,
                expand=True,
                fillcolor=(0, 0, 0, 0)
            )

        # --- Shear (needs tight crop) ---
        shear = instance.get('shear')
        if shear:
            ctx = LayerCropContext(img)
            obj = np.array(ctx.get_object()).astype(np.float32)

            sx = shear.get('x', 0.0)
            sy = shear.get('y', 0.0)

            h, w = obj.shape[:2]
            M = np.float32([
                [1,  sx, 0],
                [sy,  1, 0]
            ])
            sheared = cv2.warpAffine(
                obj, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            img = ctx.restore(Image.fromarray(sheared.astype(np.uint8), 'RGBA'))

        # --- Perspective (needs tight crop) ---
        perspective = instance.get('perspective')
        if perspective:
            ctx = LayerCropContext(img)
            obj = np.array(ctx.get_object()).astype(np.float32)

            h, w = obj.shape[:2]
            # perspective is a strength value (-1.0 to 1.0)
            # positive = tilt top inward, negative = tilt bottom inward
            strength = float(perspective) * w * 0.3

            src = np.float32([
                [0, 0], [w, 0], [w, h], [0, h]
            ])
            dst = np.float32([
                [strength, 0],
                [w - strength, 0],
                [w, h],
                [0, h]
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(
                obj, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            img = ctx.restore(Image.fromarray(warped.astype(np.uint8), 'RGBA'))

        # --- Elastic deformation (needs tight crop) ---
        elastic = instance.get('elastic')
        if elastic:
            ctx = LayerCropContext(img)
            obj = np.array(ctx.get_object()).astype(np.float32)

            h, w = obj.shape[:2]
            strength = float(elastic) * min(h, w) * 0.1

            dx = cv2.GaussianBlur(
                (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                (0, 0), sigmaX=w * 0.05
            ) * strength

            dy = cv2.GaussianBlur(
                (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                (0, 0), sigmaX=h * 0.05
            ) * strength

            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x_coords + dx).astype(np.float32)
            map_y = (y_coords + dy).astype(np.float32)

            deformed = cv2.remap(
                obj, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            img = ctx.restore(Image.fromarray(deformed.astype(np.uint8), 'RGBA'))

        # --- Crop (partial visibility) ---
        crop = instance.get('crop')
        if crop:
            # crop is a dict with sides to cut as fractions 0.0-1.0
            # e.g. {'right': 0.3} = cut 30% from right side
            arr = np.array(img)
            h, w = arr.shape[:2]

            top    = int(crop.get('top',    0.0) * h)
            bottom = int(crop.get('bottom', 0.0) * h)
            left   = int(crop.get('left',   0.0) * w)
            right  = int(crop.get('right',  0.0) * w)

            if top > 0:
                arr[:top, :, 3] = 0
            if bottom > 0:
                arr[h - bottom:, :, 3] = 0
            if left > 0:
                arr[:, :left, 3] = 0
            if right > 0:
                arr[:, w - right:, 3] = 0

            img = Image.fromarray(arr, 'RGBA')

        return img

    def _apply_appearance(self, img: Image.Image, instance: dict) -> Image.Image:
        """Opacity, blur, noise, compression."""

        # --- Opacity ---
        opacity = instance.get('opacity', 1.0)
        if opacity != 1.0:
            arr = np.array(img)
            arr[:, :, 3] = (arr[:, :, 3] * np.clip(opacity, 0, 1)).astype(np.uint8)
            img = Image.fromarray(arr, 'RGBA')

        # --- Blur (needs tight crop for clean result) ---
        blur = instance.get('blur')
        if blur:
            ctx = LayerCropContext(img)
            obj = np.array(ctx.get_object())

            blur_type   = blur.get('type', 'gaussian')
            blur_radius = max(1, int(blur.get('radius', 3)))

            if blur_type == 'gaussian':
                ksize = blur_radius * 2 + 1
                blurred = cv2.GaussianBlur(obj, (ksize, ksize), 0)

            elif blur_type == 'motion':
                angle  = blur.get('angle', 0)
                kernel = np.zeros((blur_radius, blur_radius))
                kernel[blur_radius // 2, :] = 1.0 / blur_radius
                M = cv2.getRotationMatrix2D(
                    (blur_radius // 2, blur_radius // 2), angle, 1
                )
                kernel = cv2.warpAffine(kernel, M, (blur_radius, blur_radius))
                blurred = cv2.filter2D(obj, -1, kernel)

            elif blur_type == 'defocus':
                ksize = blur_radius * 2 + 1
                kernel = np.zeros((ksize, ksize), np.float32)
                cv2.circle(kernel, (blur_radius, blur_radius), blur_radius, 1, -1)
                kernel /= kernel.sum()
                blurred = cv2.filter2D(obj, -1, kernel)

            else:
                blurred = obj

            # Preserve original alpha
            blurred[:, :, 3] = obj[:, :, 3]
            img = ctx.restore(Image.fromarray(blurred, 'RGBA'))

        # --- Noise (needs tight crop) ---
        noise = instance.get('noise')
        if noise:
            ctx = LayerCropContext(img)
            obj = np.array(ctx.get_object()).astype(np.float32)
            mask = obj[:, :, 3] > 0

            noise_type     = noise.get('type', 'gaussian')
            noise_strength = noise.get('strength', 0.1)

            if noise_type == 'gaussian':
                n = np.random.randn(*obj[:, :, :3].shape) * noise_strength * 255
                obj[:, :, :3] = np.clip(obj[:, :, :3] + n * mask[:, :, None], 0, 255)

            elif noise_type == 'salt_pepper':
                prob   = noise_strength * 0.1
                salt   = np.random.rand(*obj[:, :, :3].shape[:2]) < prob
                pepper = np.random.rand(*obj[:, :, :3].shape[:2]) < prob
                obj[:, :, :3][salt & mask]   = 255
                obj[:, :, :3][pepper & mask] = 0

            elif noise_type == 'perlin':
                # Approximate perlin with multi-scale gaussian noise
                h, w = obj.shape[:2]
                n = np.zeros((h, w))
                for scale in [4, 8, 16]:
                    small = np.random.randn(h // scale + 1, w // scale + 1)
                    n += cv2.resize(small, (w, h)) * (scale / 16.0)
                n = (n / n.std()) * noise_strength * 50
                obj[:, :, :3] = np.clip(
                    obj[:, :, :3] + n[:, :, None] * mask[:, :, None], 0, 255
                )

            img = ctx.restore(Image.fromarray(obj.astype(np.uint8), 'RGBA'))

        # --- Compression ---
        compression = instance.get('compression')
        if compression:
            ctx = LayerCropContext(img)
            obj = ctx.get_object()

            quality = max(1, min(95, int((1.0 - compression) * 95)))

            # JPEG doesn't support alpha — split, compress RGB, reattach alpha
            rgb   = obj.convert('RGB')
            alpha = obj.split()[3]

            import io
            buf = io.BytesIO()
            rgb.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            compressed_rgb = Image.open(buf).convert('RGB')

            compressed = Image.merge('RGBA', (*compressed_rgb.split(), alpha))
            img = ctx.restore(compressed)

        return img

    def _apply_color(self, img: Image.Image, instance: dict) -> Image.Image:
        """Hue, saturation, brightness, gamma, channel manipulation."""

        arr   = np.array(img)
        rgb   = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3]
        mask  = alpha > 0

        # --- Hue shift ---
        hue = instance.get('hue')
        if hue:
            shift = float(hue.get('shift', 0))
            if shift != 0:
                hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + shift / 2.0) % 180  # OpenCV hue is 0-180
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

        # --- Saturation ---
        saturation = instance.get('saturation')
        if saturation:
            scale = float(saturation.get('scale', 1.0))
            if scale != 1.0:
                hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

        # --- Brightness ---
        brightness = instance.get('brightness')
        if brightness:
            shift = float(brightness.get('shift', 0)) * 255
            rgb = np.clip(rgb + shift * mask[:, :, None], 0, 255)

        # --- Gamma ---
        gamma = instance.get('gamma')
        if gamma:
            g = float(gamma.get('value', 1.0))
            if g != 1.0:
                rgb = np.clip(255.0 * (rgb / 255.0) ** (1.0 / g), 0, 255)

        # --- Channel manipulation ---
        channel = instance.get('channel')
        if channel:
            mode = channel.get('mode')

            if mode == 'swap_rg':
                rgb = rgb[:, :, [1, 0, 2]]
            elif mode == 'swap_rb':
                rgb = rgb[:, :, [2, 1, 0]]
            elif mode == 'swap_gb':
                rgb = rgb[:, :, [0, 2, 1]]
            elif mode == 'invert':
                rgb = np.clip(255.0 - rgb, 0, 255) * mask[:, :, None]
            elif mode == 'grayscale':
                intensity = channel.get('intensity', 1.0)
                gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
                gray_rgb = np.stack([gray, gray, gray], axis=2)
                rgb = rgb * (1 - intensity) + gray_rgb * intensity

        arr = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha])
        return Image.fromarray(arr, 'RGBA')

    def _apply_compositing(self, img: Image.Image, instance: dict) -> Image.Image:
        """Edge feathering, alpha noise."""

        # --- Edge feathering ---
        edge = instance.get('edge')
        if edge:
            radius = max(1, int(edge.get('feather_radius', 3)))
            arr = np.array(img)
            alpha = arr[:, :, 3].astype(np.float32)
            alpha = cv2.GaussianBlur(alpha, (radius * 2 + 1, radius * 2 + 1), 0)
            arr[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, 'RGBA')

        # --- Alpha noise ---
        alpha_noise = instance.get('alpha_noise')
        if alpha_noise:
            strength = float(alpha_noise.get('strength', 0.1))
            arr = np.array(img)
            mask = arr[:, :, 3] > 0
            noise = (np.random.randn(*arr[:, :, 3].shape) * strength * 255 * mask)
            arr[:, :, 3] = np.clip(
                arr[:, :, 3].astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)
            img = Image.fromarray(arr, 'RGBA')

        return img

    def _synthesize_shadow(self, img: Image.Image, shadow: dict) -> Image.Image:
        """
        Synthesizes a drop shadow from the object's alpha mask.
        Returns a new RGBA image with shadow beneath the object.
        Shadow is derived entirely from the silhouette — no pre-existing shadow needed.
        """
        angle    = shadow.get('angle', 45)
        distance = shadow.get('distance', 20)
        blur_r   = max(1, shadow.get('blur_radius', 15))
        opacity  = shadow.get('opacity', 120)
        color    = shadow.get('color', (0, 0, 0))
        scale    = shadow.get('scale', 1.0)

        arr   = np.array(img)
        alpha = arr[:, :, 3]
        h, w  = alpha.shape

        # Direction from angle
        rad = np.deg2rad(angle)
        dx  = int(np.cos(rad) * distance)
        dy  = int(-np.sin(rad) * distance)

        # Build shadow layer from silhouette
        shadow_alpha = cv2.GaussianBlur(alpha, (blur_r * 2 + 1, blur_r * 2 + 1), 0)
        shadow_alpha = (shadow_alpha * (opacity / 255.0)).astype(np.uint8)

        # Optional: scale shadow (elongate / compress)
        if scale != 1.0:
            new_h = int(h * scale)
            shadow_alpha = cv2.resize(shadow_alpha, (w, new_h))
            if new_h > h:
                shadow_alpha = shadow_alpha[:h, :]
            else:
                pad = np.zeros((h - new_h, w), dtype=np.uint8)
                shadow_alpha = np.vstack([shadow_alpha, pad])

        shadow_layer = np.zeros((h, w, 4), dtype=np.uint8)
        shadow_layer[:, :, 0] = color[0]
        shadow_layer[:, :, 1] = color[1]
        shadow_layer[:, :, 2] = color[2]
        shadow_layer[:, :, 3] = shadow_alpha

        # Offset shadow by direction
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shadow_layer = cv2.warpAffine(
            shadow_layer, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Composite: shadow below, object on top
        shadow_img = Image.fromarray(shadow_layer, 'RGBA')
        result = Image.new('RGBA', img.size, (0, 0, 0, 0))
        result.paste(shadow_img, (0, 0), shadow_img)
        result.paste(img, (0, 0), img)

        return result

    def _apply_all_properties(self, img: Image.Image, instance: dict) -> Image.Image:
        """Master pipeline — applies all property groups in correct order."""
        img = self._apply_geometric(img, instance)
        img = self._apply_appearance(img, instance)
        img = self._apply_color(img, instance)
        img = self._apply_compositing(img, instance)

        # Shadow applied last — after all other transforms
        shadow = instance.get('shadow')
        if shadow and shadow.get('enabled', False):
            img = self._synthesize_shadow(img, shadow)

        return img

    # ============================================================
    # Render Final Image
    # ============================================================
    def render(self):
        # Sort layers by composite_order if set, else use list order
        layers = sorted(
            self.dataset.layers,
            key=lambda l: l['instances'][0].get('composite_order', l['index'])
        )

        result = Image.new('RGBA', self.dataset.canvas_size, (255, 255, 255, 0))

        for layer in layers:
            if not layer['enabled']:
                continue

            instance = layer['instances'][0]

            if 'edited_image' in layer:
                img = layer['edited_image']
            else:
                img = Image.open(layer['file']).convert('RGBA')

            original_size = img.size
            img = self._apply_all_properties(img, instance)

            x_offset, y_offset = instance['position']

            size_diff_x = (img.width - original_size[0]) // 2
            size_diff_y = (img.height - original_size[1]) // 2

            paste_x = x_offset - size_diff_x
            paste_y = y_offset - size_diff_y

            # Blend mode handling
            blend_mode = (instance.get('blend') or {}).get('mode', 'normal')
            if blend_mode == 'normal':
                result.paste(img, (paste_x, paste_y), img)
            else:
                result = self._apply_blend_mode(result, img, (paste_x, paste_y), blend_mode)

        rgb_result = Image.new('RGB', result.size, (255, 255, 255))
        rgb_result.paste(result, (0, 0), result)

        return rgb_result

    def _apply_blend_mode(self, base: Image.Image, layer: Image.Image, pos: tuple, mode: str) -> Image.Image:
        """Apply non-normal blend modes."""
        bx, by = pos
        base_arr  = np.array(base).astype(np.float32) / 255.0
        layer_arr = np.array(layer).astype(np.float32) / 255.0

        h, w   = layer_arr.shape[:2]
        bH, bW = base_arr.shape[:2]

        x1, y1 = max(bx, 0), max(by, 0)
        x2, y2 = min(bx + w, bW), min(by + h, bH)
        lx1 = x1 - bx
        ly1 = y1 - by

        base_region  = base_arr[y1:y2, x1:x2, :3]
        layer_region = layer_arr[ly1:ly1+(y2-y1), lx1:lx1+(x2-x1), :3]
        layer_alpha  = layer_arr[ly1:ly1+(y2-y1), lx1:lx1+(x2-x1), 3:4]

        if mode == 'multiply':
            blended = base_region * layer_region
        elif mode == 'screen':
            blended = 1 - (1 - base_region) * (1 - layer_region)
        elif mode == 'overlay':
            blended = np.where(
                base_region < 0.5,
                2 * base_region * layer_region,
                1 - 2 * (1 - base_region) * (1 - layer_region)
            )
        else:
            blended = layer_region

        composited = base_region * (1 - layer_alpha) + blended * layer_alpha
        base_arr[y1:y2, x1:x2, :3] = composited

        return Image.fromarray((base_arr * 255).astype(np.uint8), 'RGBA')

    # ============================================================
    # Visualization
    # ============================================================
    def visualize(self, save_path=None):
        layers = self.dataset.layers
        num_layers = len(layers)

        fig, axes = plt.subplots(
            1,
            num_layers + 1,
            figsize=(4 * (num_layers + 1), 4)
        )

        if num_layers == 1:
            axes = [axes[0], axes[1]]

        for i, layer in enumerate(layers):
            instance = layer['instances'][0]

            if 'edited_image' in layer:
                img = layer['edited_image']
            else:
                img = Image.open(layer['file']).convert('RGBA')

            img = self._apply_all_properties(img, instance)

            axes[i].imshow(img)

            status = "✓" if layer['enabled'] else "X"
            title  = f"{status} L{i}"
            hints  = []

            if instance['position'] != (0, 0):
                hints.append("pos")
            if instance.get('scale', 1.0) != 1.0:
                hints.append(f"{instance['scale']:.1f}x")
            if instance.get('rotation', 0) != 0:
                hints.append(f"{instance['rotation']:.0f}°")
            if instance.get('hue'):
                hints.append("hue")
            if instance.get('blur'):
                hints.append(f"blur:{instance['blur'].get('type','')}")
            if instance.get('noise'):
                hints.append(f"noise:{instance['noise'].get('type','')}")
            if instance.get('shadow', {}).get('enabled'):
                hints.append("shadow")
            if instance.get('shear'):
                hints.append("shear")
            if instance.get('elastic'):
                hints.append("elastic")
            if instance.get('perspective'):
                hints.append("persp")
            if instance.get('crop'):
                hints.append("crop")

            if hints:
                title += "\n" + ", ".join(hints)

            axes[i].set_title(title, fontsize=9)
            axes[i].axis('off')

            if not layer['enabled']:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)

        result = self.render()
        axes[-1].imshow(result)
        axes[-1].set_title("Final\nResult", fontsize=11, fontweight='bold')
        axes[-1].axis('off')

        for spine in axes[-1].spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()


# ============================================================
# Layer Dataset
# ============================================================

class LayerDataset:
    """
    Dataset containing layers and their properties.

    Responsibilities:
    - Load layer files
    - Store layer metadata and properties
    - Provide access to layers
    - Provide debugging utilities

    Does NOT:
    - Modify properties
    - Perform rendering
    """

    def __init__(self, image_id: str, layer_dir: str):
        self.image_id  = image_id
        self.layer_dir = Path(layer_dir)

        self.layer_files = sorted(
            self.layer_dir.glob(f"{image_id}-layer_*.png")
        )

        if len(self.layer_files) == 0:
            raise ValueError(f"No layer files found for {image_id}")

        self.layers = []

        for i, layer_file in enumerate(self.layer_files):
            layer = {
                'index':   i,
                'file':    layer_file,
                'type':    'background' if i == 0 else 'instance',
                'enabled': True,
                'count':   1,
                'instances': [
                    {
                        # ---- Position ----
                        'position':        (0, 0),

                        # ---- Geometric ----
                        'scale':           1.0,
                        'rotation':        0.0,
                        'flip_horizontal': False,
                        'flip_vertical':   False,
                        'shear':           None,  # {'x': float, 'y': float}
                        'perspective':     None,  # float  -1.0 to 1.0
                        'elastic':         None,  # float   0.0 to 1.0
                        'crop':            None,  # {'top': f, 'bottom': f, 'left': f, 'right': f}  fractions 0-1

                        # ---- Appearance ----
                        'opacity':         1.0,
                        'blur':            None,  # {'type': 'gaussian'|'motion'|'defocus', 'radius': int, 'angle': float}
                        'noise':           None,  # {'type': 'gaussian'|'salt_pepper'|'perlin', 'strength': float}
                        'compression':     None,  # float  0.0 (none) to 1.0 (max)

                        # ---- Color ----
                        'hue':             None,  # {'shift': float}  degrees -180 to 180
                        'saturation':      None,  # {'scale': float}  0=gray, 1=normal, 2=vivid
                        'brightness':      None,  # {'shift': float}  -1.0 to 1.0
                        'gamma':           None,  # {'value': float}  <1 darker, >1 brighter
                        'channel':         None,  # {'mode': 'swap_rg'|'swap_rb'|'swap_gb'|'invert'|'grayscale', 'intensity': float}

                        # ---- Compositing ----
                        'blend':           None,  # {'mode': 'normal'|'multiply'|'screen'|'overlay'}
                        'edge':            None,  # {'feather_radius': int}
                        'alpha_noise':     None,  # {'strength': float}
                        'composite_order': i,     # z-index, lower = further back

                        # ---- Shadow (synthesized from silhouette) ----
                        'shadow': {
                            'enabled':     False,
                            'angle':       45,        # degrees, light source direction
                            'distance':    20,        # pixel offset
                            'blur_radius': 15,        # shadow softness
                            'opacity':     120,       # 0-255
                            'color':       (0, 0, 0),
                            'scale':       1.0,       # elongate / compress shadow
                        },
                    }
                ]
            }

            self.layers.append(layer)

        bg = Image.open(self.layer_files[0])
        self.canvas_size = bg.size

        print(f"✓ Loaded {len(self.layers)} layers")
        print(f"  Canvas size: {self.canvas_size}")

    def get_layer(self, index):
        if 0 <= index < len(self.layers):
            return self.layers[index]
        return None

    def get_num_layers(self):
        return len(self.layers)

    def get_layer_file(self, index):
        if 0 <= index < len(self.layers):
            return self.layers[index]['file']
        return None

    def get_duplicates_of(self, index):
        if not 0 <= index < len(self.layers):
            return []
        base_file = self.layers[index]['file']
        return [
            i for i, layer in enumerate(self.layers)
            if layer['file'] == base_file
        ]

    def describe(self):
        print("\nLayerDataset Structure")
        print("=" * 80)
        print(f"Image ID      : {self.image_id}")
        print(f"Layer Dir     : {self.layer_dir}")
        print(f"Canvas Size   : {self.canvas_size}")
        print(f"Total Layers  : {len(self.layers)}")

        print("\nLayers")
        print("-" * 80)

        for layer in self.layers:
            print(f"\nLayer {layer['index']}")
            print(f"  File    : {layer['file'].name}")
            print(f"  Type    : {layer['type']}")
            print(f"  Enabled : {layer['enabled']}")
            print(f"  Count   : {layer['count']}")
            print("  Instances")
            for inst_id, instance in enumerate(layer['instances']):
                print(f"    Instance {inst_id}")
                for key, value in instance.items():
                    print(f"      {key:20s}: {value}")
