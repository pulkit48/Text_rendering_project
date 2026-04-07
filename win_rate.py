from pathlib import Path
from PIL import Image, ImageDraw
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Object Region استخراج
# =========================
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


# =========================
# Dataset
# =========================
class LayerDataset:
    def __init__(self, image_id: str, layer_dir: str):
        self.image_id = image_id
        self.layer_dir = Path(layer_dir)

        self.layer_files = sorted(
            self.layer_dir.glob(f"{image_id}_layer_*.png")
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

        bg = Image.open(self.layer_files[0])
        self.canvas_size = bg.size

        print(f"✓ Loaded {len(self.layers)} layers")
        print(f"Canvas size: {self.canvas_size}")

    # =========================
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

    # =========================
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

    def get_object_mask(self, index):
        return self.layers[index]['instances'][0].get('object_mask')

    # =========================
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
            print(f"\nLayer {layer.get('index')}")

            for key, value in layer.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                elif isinstance(value, list):
                    print(f"  {key}:")
                    for i, item in enumerate(value):
                        print(f"    [{i}] {item}")
                else:
                    print(f"  {key}: {value}")

        print("=" * 80)


# =========================
# Renderer
# =========================
class LayerRenderer:
    def __init__(self, dataset):
        self.dataset = dataset

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

        return img

    # =========================
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

    # =========================
    def visualize(self):
        result = self.render()

        plt.imshow(result)
        plt.title("Final Result")
        plt.axis('off')
        plt.show()


# =========================
# Editor
# =========================
class LayerEditor:
    def __init__(self, dataset):
        self.dataset = dataset

    def set_enabled(self, index, enabled: bool):
        layers = self.dataset.layers
        if 0 <= index < len(layers):
            layers[index]['enabled'] = enabled
            print(f"✓ Layer {index} {'enabled' if enabled else 'disabled'}")

    def set_position(self, index: int, x: int, y: int):
        layers = self.dataset.layers
        if 0 <= index < len(layers):
            layers[index]['instances'][0]['position'] = (x, y)
            print(f"✓ Layer {index} position: ({x}, {y})")

    def set_scale(self, index: int, scale: float):
        layers = self.dataset.layers
        if 0 <= index < len(layers):
            layers[index]['instances'][0]['scale'] = scale
            print(f"✓ Layer {index} scale: {scale}x")

    def set_rotation(self, index: int, degrees: float):
        layers = self.dataset.layers
        if 0 <= index < len(layers):
            layers[index]['instances'][0]['rotation'] = degrees
            print(f"✓ Layer {index} rotation: {degrees}°")
