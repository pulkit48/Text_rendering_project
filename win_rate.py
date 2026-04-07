from pathlib import Path
from PIL import Image
import numpy as np
import cv2


# ==========================================
# ObjectRegion
# ==========================================
class ObjectRegion:
    def __init__(self, layer):
        img = Image.open(layer['file']).convert('RGBA')
        arr = np.array(img)

        # Alpha mask
        mask = arr[:, :, 3] > 0

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        ys = np.where(rows)[0]
        xs = np.where(cols)[0]

        # No object case
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

        # Load layer files
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
                    'image': None
                }]
            }

            self.layers.append(layer)

        # Canvas size
        bg = Image.open(self.layer_files[0])
        self.canvas_size = bg.size

        print(f"✓ Loaded {len(self.layers)} layers")
        print(f"Canvas size: {self.canvas_size}")

    # ==========================================
    # Preprocess (extract objects)
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
                inst['image'] = Image.fromarray(region.object, 'RGBA')

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

        return [
            i for i, layer in enumerate(self.layers)
            if layer['file'] == base_file
        ]

    def get_object_mask(self, index):
        return self.layers[index]['instances'][0].get('object_mask')

    # ==========================================
    # EDGE MASK (IMPORTANT — you were missing this)
    # ==========================================
    def get_edge_mask(self, index, thickness=10):
        mask = self.layers[index]['instances'][0].get('object_mask')

        if mask is None:
            return None

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

                elif hasattr(value, 'shape'):  # numpy
                    print(f"  {key}: ndarray shape={value.shape} dtype={value.dtype}")

                elif hasattr(value, 'size') and hasattr(value, 'mode'):  # PIL
                    print(f"  {key}: PIL.Image size={value.size} mode={value.mode}")

                else:
                    print(f"  {key}: {value}")

        print("=" * 80)

class LayerRenderer:
    def __init__(self, dataset):
        self.dataset = dataset

    # ==========================================
    def _apply_all_properties(self, img, instance):

        # Flip
        if instance['flip_horizontal']:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if instance['flip_vertical']:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Scale
        if instance['scale'] != 1.0:
            new_w = int(img.width * instance['scale'])
            new_h = int(img.height * instance['scale'])
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Rotation
        if instance['rotation'] != 0:
            img = img.rotate(
                instance['rotation'],
                resample=Image.BICUBIC,
                expand=True,
                fillcolor=(0, 0, 0, 0)
            )

        # Color transforms
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

                rgb = (
                    rgb * (1 - intensity) +
                    rgb_new * intensity
                ).astype(np.uint8)

            elif mode == "warmer":
                shift = np.array([30, 10, -20], dtype=np.float32)
                rgb = np.clip(
                    rgb.astype(np.float32) + shift * intensity * 2,
                    0, 255
                ).astype(np.uint8)

            elif mode == "cooler":
                shift = np.array([-20, 0, 30], dtype=np.float32)
                rgb = np.clip(
                    rgb.astype(np.float32) + shift * intensity * 2,
                    0, 255
                ).astype(np.uint8)

            img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")

        return img

    # ==========================================
    def render(self):
        result = Image.new(
            'RGBA',
            self.dataset.canvas_size,
            (255, 255, 255, 0)
        )

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

            img = instance['image']
            img = self._apply_all_properties(img, instance)

            x1, y1, _, _ = instance['bbox']
            dx, dy = instance['position']

            result.paste(img, (x1 + dx, y1 + dy), img)

        rgb_result = Image.new('RGB', result.size, (255, 255, 255))
        rgb_result.paste(result, (0, 0), result)

        return rgb_result

    # ==========================================
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

            if layer['type'] == 'background':
                img = Image.open(layer['file']).convert('RGBA')
            else:
                if instance['image'] is None:
                    axes[i].set_title(f"L{i}\n<empty>")
                    axes[i].axis('off')
                    continue
                img = instance['image']

            img = self._apply_all_properties(img, instance)

            axes[i].imshow(img)

            status = "✓" if layer['enabled'] else "✗"
            title = f"{status} L{i}"
            axes[i].set_title(title)
            axes[i].axis('off')

        result = self.render()

        axes[-1].imshow(result)
        axes[-1].set_title("Final Result", fontweight='bold')
        axes[-1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

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
    # Duplication
    def set_count(self, index: int, count: int):
        import copy
        layers = self.dataset.layers

        if not (0 <= index < len(layers)):
            print(f"✗ Invalid layer index: {index}")
            return

        base_layer = layers[index]
        base_file = base_layer['file']
        base_inst = base_layer['instances'][0]

        current_indices = [
            i for i, l in enumerate(layers)
            if l['file'] == base_file
        ]

        current_count = len(current_indices)

        if count > current_count:
            for _ in range(count - current_count):
                new_inst = copy.deepcopy(base_inst)

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
            keep = current_indices[:count]
            layers[:] = [
                l for i, l in enumerate(layers)
                if i in keep or l['file'] != base_file
            ]

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

    # ==========================================
    # REMOVE RANDOM PART  (FROM IMAGE)
    def remove_random_part(self, index: int, visualize=True):
        import matplotlib.pyplot as plt

        layer = self.dataset.layers[index]

        if layer['instances'][0]['image'] is None:
            print("✗ No image found")
            return

        original_img = np.array(layer['instances'][0]['image'], dtype=np.uint8)

        alpha = original_img[:, :, 3] > 0
        mask = self._generate_structured_mask(alpha)

        edited = original_img.copy()
        edited[mask] = [0, 0, 0, 0]

        edited_img = Image.fromarray(edited, "RGBA")

        layer['instances'][0]['image'] = edited_img

        print(f"✓ Layer {index} random part removed")

        if visualize:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(original_img)
            axes[0].set_title("Original")
            axes[0].axis('off')

            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title("Mask")
            axes[1].axis('off')

            axes[2].imshow(edited_img)
            axes[2].set_title("Edited")
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()

    # ==========================================
    # EDGE DISTORTION (FROM YOUR IMAGE)
    def distort_object_edge(self, index: int,
                            max_shift: int = 3,
                            probability: float = 0.7,
                            visualize: bool = False):

        layers = self.dataset.layers

        if not (0 <= index < len(layers)):
            print("✗ Invalid layer index")
            return

        layer = layers[index]

        if layer['type'] == 'background':
            print("✗ Cannot distort background")
            return

        if layer['instances'][0]['image'] is None:
            print("✗ No object image")
            return

        img = np.array(layer['instances'][0]['image'], dtype=np.uint8)
        H, W, _ = img.shape

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

            ny = np.clip(y + dy, 0, H - 1)
            nx = np.clip(x + dx, 0, W - 1)

            distorted[ny, nx] = img[y, x]
            distorted[y, x] = [0, 0, 0, 0]

        layer['instances'][0]['image'] = Image.fromarray(distorted, "RGBA")

        print(f"✓ Edge distortion applied to layer {index}")

        if visualize:
            import matplotlib.pyplot as plt
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
    # RESET SINGLE LAYER
    def reset_layer(self, index: int):
        layers = self.dataset.layers

        if 0 <= index < len(layers):
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
                    'image': layers[index]['instances'][0].get('image'),
                    'bbox': layers[index]['instances'][0].get('bbox')
                }]
            })

            print(f"✓ Layer {index} reset")

    # ==========================================
    # RESET ALL
    def reset_all(self):
        for i in range(len(self.dataset.layers)):
            self.reset_layer(i)

    # ==========================================
    # STATUS (FROM YOUR IMAGE)
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

            props = []
            props.append(f"pos={instance['position']}")
            props.append(f"scale={instance['scale']:.2f}")
            props.append(f"rot={instance['rotation']:.1f}")
            props.append(f"flip_h={instance['flip_horizontal']}")
            props.append(f"flip_v={instance['flip_vertical']}")
            props.append(f"color={instance['color_mode']}")
            props.append(f"intensity={instance['color_intensity']:.1f}")

            props_str = ", ".join(props)

            dup_mark = " [DUPLICATE]" if is_duplicate else ""

            print(f"{status} Layer {i:2d} ({layer_type:10s}): {props_str}{dup_mark}")

        print("=" * 80)

    # ==========================================
    # INTERNAL MASK (FROM YOUR CODE)
    def _generate_structured_mask(self, alpha):
        H, W = alpha.shape

        ys, xs = np.where(alpha)
        if len(ys) == 0:
            return np.zeros_like(alpha)

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        mask = np.zeros_like(alpha)

        shape_type = random.choice(["ellipse", "blob", "cut"])

        if shape_type == "ellipse":
            temp = np.zeros_like(alpha, dtype=np.uint8)

            center = (
                random.randint(x_min, x_max),
                random.randint(y_min, y_max)
            )

            axes = (
                random.randint(10, max(10, (x_max - x_min)//3)),
                random.randint(10, max(10, (y_max - y_min)//3))
            )

            cv2.ellipse(temp, center, axes, 0, 0, 360, 1, -1)
            mask = temp.astype(bool)

        elif shape_type == "blob":
            noise = np.random.rand(H, W)
            blob = (noise > 0.8).astype(np.uint8)
            blob = cv2.GaussianBlur(blob.astype(float), (15, 15), 0)
            mask = blob > 0.4

        else:
            thickness = random.randint(5, 20)
            if random.random() < 0.5:
                x = random.randint(x_min, x_max)
                mask[:, max(0, x-thickness):min(W, x+thickness)] = 1
            else:
                y = random.randint(y_min, y_max)
                mask[max(0, y-thickness):min(H, y+thickness), :] = 1

            mask = mask.astype(bool)

        return mask & alpha
