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
