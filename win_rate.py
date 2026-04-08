from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import random
import copy


# ==========================================
# ObjectRegion
# ==========================================
class ObjectRegion:
    def __init__(self, layer):
        img = Image.open(layer['file']).convert('RGBA')
        arr = np.array(img)

        mask = arr[:, :, 3] > 0

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        ys = np.where(rows)[0]
        xs = np.where(cols)[0]

        if len(ys) == 0 or len(xs) == 0:
            self.bbox = None
            self.object = None
            self.object_mask = None
            return

        y1, y2 = ys[[0, -1]]
        x1, x2 = xs[[0, -1]]

        self.bbox = (x1, y1, x2 + 1, y2 + 1)
        self.object = arr[y1:y2+1, x1:x2+1]
        self.object_mask = self.object[:, :, 3] > 0


# ==========================================
# LayerDataset
# ==========================================
class LayerDataset:
    def __init__(self, image_id: str, layer_dir: str):
        self.image_id = image_id
        self.layer_dir = Path(layer_dir)

        self.layer_files = sorted(
            self.layer_dir.glob(f"{image_id}-layer_*.png")
        )

        if len(self.layer_files) == 0:
            raise ValueError(f"No layer files found for {image_id}")

        self.layers = []

        for i, layer_file in enumerate(self.layer_files):
            layer = {
                'index': i,
                'file': layer_file,
                'type': 'background' if i == 0 else 'instance',
                'enabled': True,
                'count': 1,
                'instances': [{
                    'position': (0, 0),
                    'scale': 1.0,
                    'rotation': 0.0,
                    'flip_horizontal': False,
                    'flip_vertical': False,
                    'color_mode': None,
                    'color_intensity': 0.5,
                    'saturation_boost': None,
                    'brightness_adjust': None,
                    'bbox': None,
                    'object_mask': None,
                    # BUG 8 FIX: store original_image separately so edits are
                    # always reversible and reset_layer can restore a clean copy
                    'original_image': None,
                    'image': None
                }]
            }
            self.layers.append(layer)

        bg = Image.open(self.layer_files[0])
        self.canvas_size = bg.size

        print(f"✓ Loaded {len(self.layers)} layers")
        print(f"Canvas size: {self.canvas_size}")

    # ==========================================
    # Preprocess
    # ==========================================
    def preprocess(self):
        for layer in self.layers:
            if layer['type'] == 'background':
                continue

            region = ObjectRegion(layer)

            inst = layer['instances'][0]
            inst['bbox'] = region.bbox
            inst['object_mask'] = region.object_mask

            if region.object is not None:
                pil_img = Image.fromarray(region.object, 'RGBA')
                # BUG 8 FIX: keep an untouched original so reset can always
                # restore a clean copy regardless of how many edits were made
                inst['original_image'] = pil_img.copy()
                inst['image'] = pil_img

    # ==========================================
    # Basic getters
    # ==========================================
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
        if not (0 <= index < len(self.layers)):
            return []
        base_file = self.layers[index]['file']
        return [i for i, layer in enumerate(self.layers) if layer['file'] == base_file]

    def get_object_mask(self, index):
        return self.layers[index]['instances'][0].get('object_mask')

    # ==========================================
    # BUG 2 FIX: edge mask is derived from the LIVE instance image, not the
    # stale bbox-time object_mask.  This ensures it always matches the current
    # image dimensions even after scale / rotation / flip.
    # ==========================================
    def get_edge_mask(self, index, thickness=10):
        inst = self.layers[index]['instances'][0]

        if inst.get('image') is None:
            return None

        # Recompute mask from the current live image
        arr = np.array(inst['image'])
        mask = arr[:, :, 3] > 0

        kernel = np.ones((thickness, thickness), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel)
        edge = mask.astype(np.uint8) - eroded

        return edge.astype(bool)

    # ==========================================
    # Debug print
    # ==========================================
    def describe(self):
        print("\nLayerDataset Structure")
        print("=" * 80)
        print(f"Image ID   : {self.image_id}")
        print(f"Layer Dir  : {self.layer_dir}")
        print(f"Canvas Size: {self.canvas_size}")
        print(f"Total Layers: {len(self.layers)}")
        print("\nLayers")
        print("-" * 80)

        for layer in self.layers:
            print(f"\nLayer {layer.get('index', '?')}")
            for key, value in layer.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                elif isinstance(value, list):
                    print(f"  {key}:")
                    for i, item in enumerate(value):
                        print(f"    [{i}]")
                        if isinstance(item, dict):
                            for k, v in item.items():
                                print(f"      {k}: {v}")
                        else:
                            print(f"      {item}")
                elif hasattr(value, 'shape'):
                    print(f"  {key}: ndarray shape={value.shape} dtype={value.dtype}")
                elif hasattr(value, 'size') and hasattr(value, 'mode'):
                    print(f"  {key}: PIL.Image size={value.size} mode={value.mode}")
                else:
                    print(f"  {key}: {value}")

        print("=" * 80)


# ==========================================
# LayerRenderer
# ==========================================
class LayerRenderer:
    def __init__(self, dataset):
        self.dataset = dataset

    # ==========================================
    def _apply_all_properties(self, img, instance):
        if instance['flip_horizontal']:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if instance['flip_vertical']:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if instance['scale'] != 1.0:
            new_w = int(img.width * instance['scale'])
            new_h = int(img.height * instance['scale'])
            img = img.resize((new_w, new_h), Image.LANCZOS)

        if instance['rotation'] != 0:
            img = img.rotate(
                instance['rotation'],
                resample=Image.BICUBIC,
                expand=True,
                fillcolor=(0, 0, 0, 0)
            )

        if instance['color_mode']:
            img_array = np.array(img)
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
            mode = instance['color_mode']
            intensity = instance['color_intensity']

            if mode == "grayscale":
                gray = (
                    0.299 * rgb[:, :, 0] +
                    0.587 * rgb[:, :, 1] +
                    0.114 * rgb[:, :, 2]
                )
                rgb_new = np.stack([gray, gray, gray], axis=2)
                rgb = (rgb * (1 - intensity) + rgb_new * intensity).astype(np.uint8)

            elif mode == "warmer":
                shift = np.array([30, 10, -20], dtype=np.float32)
                rgb = np.clip(
                    rgb.astype(np.float32) + shift * intensity * 2, 0, 255
                ).astype(np.uint8)

            elif mode == "cooler":
                shift = np.array([-20, 0, 30], dtype=np.float32)
                rgb = np.clip(
                    rgb.astype(np.float32) + shift * intensity * 2, 0, 255
                ).astype(np.uint8)

            img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")

        return img

    # ==========================================
    # BUG 6 FIX: after scale/rotation the image size changes, so we can no
    # longer blindly paste at (x1, y1).  We compute the size delta and shift
    # the paste origin so the object stays visually centred on its original
    # bbox position.
    # ==========================================
    def render(self):
        result = Image.new('RGBA', self.dataset.canvas_size, (255, 255, 255, 0))

        for layer in self.dataset.layers:
            if not layer['enabled']:
                continue

            instance = layer['instances'][0]

            if layer['type'] == 'background':
                img = Image.open(layer['file']).convert('RGBA')
                img = self._apply_all_properties(img, instance)
                result.paste(img, (0, 0), img)
                continue

            if instance['image'] is None:
                continue

            x1, y1, x2, y2 = instance['bbox']
            orig_w = x2 - x1
            orig_h = y2 - y1

            img = instance['image'].copy()
            img = self._apply_all_properties(img, instance)

            dx, dy = instance['position']

            # Offset so the transformed image is centred on the original bbox
            size_dx = (img.width  - orig_w) // 2
            size_dy = (img.height - orig_h) // 2

            paste_x = x1 + dx - size_dx
            paste_y = y1 + dy - size_dy

            result.paste(img, (paste_x, paste_y), img)

        rgb_result = Image.new('RGB', result.size, (255, 255, 255))
        rgb_result.paste(result, (0, 0), result)
        return rgb_result

    # ==========================================
    # BUG 7 FIX: individual layer panels must NOT call _apply_all_properties
    # because render() already applies transforms for the final panel.
    # Show each layer's raw instance image / background as-is.
    # ==========================================
    def visualize(self, save_path=None):
        import matplotlib.pyplot as plt

        layers = self.dataset.layers
        num_layers = len(layers)

        fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))

        if num_layers == 1:
            axes = [axes[0], axes[1]]

        for i, layer in enumerate(layers):
            instance = layer['instances'][0]

            if layer['type'] == 'background':
                # Show background without any transforms applied
                img = Image.open(layer['file']).convert('RGBA')
            else:
                if instance['image'] is None:
                    axes[i].set_title(f"L{i}\n<empty>")
                    axes[i].axis('off')
                    continue
                # Show the raw stored image — do NOT re-apply transforms here
                img = instance['image']

            axes[i].imshow(img)
            status = "✓" if layer['enabled'] else "✗"
            axes[i].set_title(f"{status} L{i}")
            axes[i].axis('off')

        # Final panel uses render() which correctly applies all transforms once
        result = self.render()
        axes[-1].imshow(result)
        axes[-1].set_title("Final Result", fontweight='bold')
        axes[-1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()


# ==========================================
# LayerEditor
# ==========================================
class LayerEditor:
    def __init__(self, dataset):
        self.dataset = dataset

    # ==========================================
    # Enable / Disable
    def set_enabled(self, index: int, enabled: bool):
        layers = self.dataset.layers
        if 0 <= index < len(layers):
            layers[index]['enabled'] = enabled
            print(f"✓ Layer {index} {'enabled' if enabled else 'disabled'}")

    # ==========================================
    # BUG 3 FIX: new duplicate instances must start from clean defaults, not a
    # deep-copy of a potentially mutated base instance.  Only the structural
    # fields (bbox, object_mask, original_image) are carried over — everything
    # else is reset to factory defaults so each duplicate is independent.
    # ==========================================
    def set_count(self, index: int, count: int):
        layers = self.dataset.layers

        if not (0 <= index < len(layers)):
            print(f"✗ Invalid layer index: {index}")
            return

        base_layer = layers[index]
        base_file = base_layer['file']
        base_inst = base_layer['instances'][0]

        current_indices = [i for i, l in enumerate(layers) if l['file'] == base_file]
        current_count = len(current_indices)

        if count > current_count:
            for _ in range(count - current_count):
                # Start from clean defaults; only copy immutable source data
                new_inst = {
                    'position': (0, 0),
                    'scale': 1.0,
                    'rotation': 0.0,
                    'flip_horizontal': False,
                    'flip_vertical': False,
                    'color_mode': None,
                    'color_intensity': 0.5,
                    'saturation_boost': None,
                    'brightness_adjust': None,
                    'bbox': base_inst.get('bbox'),
                    'object_mask': base_inst.get('object_mask'),
                    'original_image': base_inst.get('original_image'),
                    # Give each duplicate its own clean image copy
                    'image': (
                        base_inst['original_image'].copy()
                        if base_inst.get('original_image') is not None
                        else None
                    ),
                }

                new_layer = {
                    'index': len(layers),
                    'file': base_file,
                    'type': base_layer['type'],
                    'enabled': True,
                    'count': 1,
                    'instances': [new_inst]
                }
                layers.append(new_layer)

        elif count < current_count:
            keep = set(current_indices[:count])
            layers[:] = [l for i, l in enumerate(layers) if i in keep or l['file'] != base_file]
            for i, l in enumerate(layers):
                l['index'] = i

        print(f"✓ Layer {index} now has {count} instances")

    # ==========================================
    # Spatial transforms
    def set_position(self, index: int, x: int, y: int):
        self.dataset.layers[index]['instances'][0]['position'] = (x, y)

    def set_scale(self, index: int, scale: float):
        self.dataset.layers[index]['instances'][0]['scale'] = scale

    def set_rotation(self, index: int, degrees: float):
        self.dataset.layers[index]['instances'][0]['rotation'] = degrees

    # ==========================================
    # Flip
    def set_flip_horizontal(self, index: int, flip: bool):
        self.dataset.layers[index]['instances'][0]['flip_horizontal'] = flip

    def set_flip_vertical(self, index: int, flip: bool):
        self.dataset.layers[index]['instances'][0]['flip_vertical'] = flip

    # ==========================================
    # Color
    def set_color(self, index: int, mode: str,
                  intensity: float = 0.5,
                  saturation_boost=None,
                  brightness_adjust=None):
        inst = self.dataset.layers[index]['instances'][0]
        inst['color_mode'] = mode
        inst['color_intensity'] = intensity
        inst['saturation_boost'] = saturation_boost
        inst['brightness_adjust'] = brightness_adjust

    # =====================================
    def distort_object_edge(self, index: int,
                            max_shift: int = 3,
                            probability: float = 0.7,
                            visualize: bool = False):
        import matplotlib.pyplot as plt

        layers = self.dataset.layers

        if not (0 <= index < len(layers)):
            print("✗ Invalid layer index")
            return

        layer = layers[index]

        if layer['type'] == 'background':
            print("✗ Cannot distort background")
            return

        inst = layer['instances'][0]

        if inst['image'] is None:
            print("✗ No object image")
            return

        img = np.array(inst['image'], dtype=np.uint8)
        H, W, _ = img.shape

        # get_edge_mask is now computed from the live image (Bug 2 fix)
        edge_mask = self.dataset.get_edge_mask(index, thickness=2)

        if edge_mask is None:
            print("✗ No edge mask available")
            return

        distorted = img.copy()
        ys, xs = np.where(edge_mask)

        for y, x in zip(ys, xs):
            if random.random() > probability:
                continue

            dy = random.randint(-max_shift, max_shift)
            dx = random.randint(-max_shift, max_shift)

            # BUG 4 FIX: skip zero-shift — clearing src with no move erases pixel
            if dy == 0 and dx == 0:
                continue

            ny = int(np.clip(y + dy, 0, H - 1))
            nx = int(np.clip(x + dx, 0, W - 1))

            # Only move pixel outward (destination was transparent in original)
            # This prevents overwriting another edge pixel that was shifted here
            if img[ny, nx, 3] == 0:
                distorted[ny, nx] = img[y, x]
                distorted[y, x] = [0, 0, 0, 0]

        inst['image'] = Image.fromarray(distorted, "RGBA")
        print(f"✓ Edge distortion applied to layer {index}")

        if visualize:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img)
            axes[0].set_title("Before")
            axes[0].axis('off')
            axes[1].imshow(distorted)
            axes[1].set_title("After")
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()

    # ==========================================
    # BUG 1 FIX: reset_layer reloads a fresh image from original_image instead
    # of recycling whatever mutated image currently sits in the instance.
    # ==========================================
    def reset_layer(self, index: int):
        layers = self.dataset.layers

        if not (0 <= index < len(layers)):
            return

        layer = layers[index]
        inst = layer['instances'][0]

        original_image = inst.get('original_image')

        layers[index].update({
            'enabled': True,
            'count': 1,
            'instances': [{
                'position': (0, 0),
                'scale': 1.0,
                'rotation': 0.0,
                'flip_horizontal': False,
                'flip_vertical': False,
                'color_mode': None,
                'color_intensity': 0.5,
                'saturation_boost': None,
                'brightness_adjust': None,
                'bbox': inst.get('bbox'),
                'object_mask': inst.get('object_mask'),
                'original_image': original_image,
                # BUG 1 FIX: always restore from original, never from current
                'image': original_image.copy() if original_image is not None else None,
            }]
        })

        print(f"✓ Layer {index} reset")

    # ==========================================
    # Reset all
    def reset_all(self):
        for i in range(len(self.dataset.layers)):
            self.reset_layer(i)

    # ==========================================
    # Status
    def show_status(self):
        layers = self.dataset.layers
        print("\nLayer Properties")
        print("=" * 80)

        seen_files = {}
        for layer in layers:
            file_key = str(layer['file'])
            if file_key not in seen_files:
                seen_files[file_key] = []
            seen_files[file_key].append(layer)

        for layer in layers:
            i = layer['index']
            instance = layer['instances'][0]
            status = "✓" if layer['enabled'] else "✗"
            layer_type = layer['type'].capitalize()
            duplicates = seen_files[str(layer['file'])]
            is_duplicate = len(duplicates) > 1 and layer != duplicates[0]

            props = [
                f"pos={instance['position']}",
                f"scale={instance['scale']:.2f}",
                f"rot={instance['rotation']:.1f}",
                f"flip_h={instance['flip_horizontal']}",
                f"flip_v={instance['flip_vertical']}",
                f"color={instance['color_mode']}",
                f"intensity={instance['color_intensity']:.1f}",
            ]

            dup_mark = " [DUPLICATE]" if is_duplicate else ""
            print(f"{status} Layer {i:2d} ({layer_type:10s}): {', '.join(props)}{dup_mark}")

        print("=" * 80)

    # ==========================================
    # BUG 5 FIX: blob branch is now seeded entirely within the object bounding
    # box so the random noise is guaranteed to land on visible 
    def remove_random_part(self, index: int, mode=None, visualize=True):
    import matplotlib.pyplot as plt

    layer = self.dataset.layers[index]
    inst = layer['instances'][0]

    if inst['image'] is None:
        return

    img = np.array(inst['image'], dtype=np.uint8)
    alpha = img[:, :, 3] > 0

    ys, xs = np.where(alpha)
    if len(ys) == 0:
        return

    obj_pixels = list(zip(ys, xs))
    total_pixels = len(obj_pixels)

    if mode is None:
        mode = random.choice(["solid", "split"])

    mask = np.zeros_like(alpha, dtype=bool)

    if mode == "solid":
        sy, sx = random.choice(obj_pixels)

        radius = random.randint(20, 50)

        y_min = max(0, sy - radius)
        y_max = min(alpha.shape[0], sy + radius)
        x_min = max(0, sx - radius)
        x_max = min(alpha.shape[1], sx + radius)

        h = y_max - y_min
        w = x_max - x_min

        noise = np.random.rand(h, w)
        noise = cv2.GaussianBlur(noise, (21, 21), 0)

        blob = noise > 0.5

        mask[y_min:y_max, x_min:x_max] = blob

    else:
        num_seeds = random.randint(2, 5)
        seeds = random.sample(obj_pixels, num_seeds)

        for sy, sx in seeds:
            radius = random.randint(10, 30)

            y_min = max(0, sy - radius)
            y_max = min(alpha.shape[0], sy + radius)
            x_min = max(0, sx - radius)
            x_max = min(alpha.shape[1], sx + radius)

            h = y_max - y_min
            w = x_max - x_min

            noise = np.random.rand(h, w)
            noise = cv2.GaussianBlur(noise, (11, 11), 0)

            blob = noise > 0.5

            mask[y_min:y_max, x_min:x_max] |= blob

    mask = mask & alpha

    if mask.sum() == 0:
        y, x = random.choice(obj_pixels)
        mask[max(0, y-5):y+5, max(0, x-5):x+5] = True

    if mask.sum() > 0.5 * total_pixels:
        mask = mask & (np.random.rand(*mask.shape) > 0.5)

    edited = img.copy()
    edited[mask] = [0, 0, 0, 0]

    inst['image'] = Image.fromarray(edited, "RGBA")

    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img); axes[0].axis('off')
        axes[1].imshow(mask, cmap='gray'); axes[1].axis('off')
        axes[2].imshow(edited); axes[2].axis('off')
        plt.tight_layout()
        plt.show()
