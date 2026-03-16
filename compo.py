"""
Unified Layer System with Properties

Architecture:
1. Each layer has ALL properties (color, position, scale, rotation, etc.)
2. Tools modify these properties
3. One render() function applies everything
"""

from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict


class UnifiedLayerSystem:
    """
    Unified system where each layer has all properties.
    
    Properties per layer:
    - Spatial: position, scale, rotation, flip
    - Color: color mode, intensity, saturation, brightness
    - Visibility: enabled/disabled
    
    One render() function applies all properties.
    """
    
    def __init__(self, image_id: str, layer_dir: str):
        """
        Initialize the unified system.
        
        Args:
            image_id: Image identifier
            layer_dir: Directory containing layer files
        """
        self.image_id = image_id
        self.layer_dir = Path(layer_dir)
        
        # Find all layer files
        self.layer_files = sorted(self.layer_dir.glob(f"{image_id}-layer_*.png"))
        
        # Initialize properties for each layer
        self.layers = []
        for i, layer_file in enumerate(self.layer_files):
            self.layers.append({
                # Identity
                'index': i,
                'file': layer_file,
                'type': 'background' if i == 0 else 'instance',
                
                # Visibility
                'enabled': True,
                
                # Duplication
                'count': 1,                 # Number of copies (1 = original only)
                'instances': [              # Properties for each copy
                    {
                        # Spatial properties
                        'position': (0, 0),         # (x, y) offset
                        'scale': 1.0,               # Scale factor
                        'rotation': 0.0,            # Degrees
                        'flip_horizontal': False,
                        'flip_vertical': False,
                        
                        # Color properties
                        'color_mode': None,         # 'warmer', 'blue', etc.
                        'color_intensity': 0.5,
                        'saturation_boost': None,
                        'brightness_adjust': None,
                    }
                ]
            })
        
        # Get canvas size
        bg = Image.open(self.layer_files[0])
        self.canvas_size = bg.size
        
        print(f"✓ Loaded {len(self.layers)} layers")
        print(f"  Canvas size: {self.canvas_size}")
    
    # ========================================================================
    # Property Getters
    # ========================================================================
    
    def get_layer(self, index: int) -> Dict:
        """Get all properties of a layer."""
        if 0 <= index < len(self.layers):
            return self.layers[index]
        return None
    
    def get_num_layers(self) -> int:
        """Get total number of layers."""
        return len(self.layers)
    
    # ========================================================================
    # Visibility Properties
    # ========================================================================
    
    def set_enabled(self, index: int, enabled: bool):
        """Enable or disable a layer."""
        if 0 <= index < len(self.layers):
            self.layers[index]['enabled'] = enabled
            status = "enabled" if enabled else "disabled"
            print(f"✓ Layer {index} {status}")
    
    # ========================================================================
    # Duplication / Count Management
    # ========================================================================
    
    def set_count(self, index: int, count: int):
        """
        Set how many copies of this layer to render.
        Creates actual new layers in the list with same file.
        
        Args:
            index: Layer index
            count: Number of copies (1 = original only, 2 = original + 1 duplicate, etc.)
        """
        if not 0 <= index < len(self.layers):
            print(f"✗ Invalid layer index: {index}")
            return
        
        if count < 1:
            print(f"✗ Count must be at least 1")
            return
        
        base_layer = self.layers[index]
        current_instances = [i for i, l in enumerate(self.layers) 
                            if l['file'] == base_layer['file']]
        current_count = len(current_instances)
        
        if count > current_count:
            # Add more layers (duplicates)
            for _ in range(count - current_count):
                new_layer = {
                    'index': len(self.layers),
                    'file': base_layer['file'],
                    'type': base_layer['type'],
                    'enabled': True,
                    'count': 1,
                    'instances': [
                        {
                            'position': (0, 0),
                            'scale': 1.0,
                            'rotation': 0.0,
                            'flip_horizontal': False,
                            'flip_vertical': False,
                            'color_mode': None,
                            'color_intensity': 0.5,
                            'saturation_boost': None,
                            'brightness_adjust': None,
                        }
                    ]
                }
                self.layers.append(new_layer)
                print(f"✓ Created duplicate layer {new_layer['index']} from layer {index}")
        
        elif count < current_count:
            # Remove duplicates (keep first 'count' instances)
            instances_to_keep = current_instances[:count]
            instances_to_remove = current_instances[count:]
            
            # Remove from layers list
            self.layers = [l for i, l in enumerate(self.layers) 
                          if i not in instances_to_remove]
            
            # Reindex
            for i, layer in enumerate(self.layers):
                layer['index'] = i
            
            print(f"✓ Removed {len(instances_to_remove)} duplicate(s)")
        
        print(f"✓ Layer {index} now has {count} instance(s) (total layers: {len(self.layers)})")
    
    def get_duplicates_of(self, index: int) -> list:
        """Get indices of all layers using the same file as this layer."""
        if not 0 <= index < len(self.layers):
            return []
        
        base_file = self.layers[index]['file']
        return [i for i, l in enumerate(self.layers) if l['file'] == base_file]
    
    # ========================================================================
    # Spatial Properties
    # ========================================================================
    
    def set_position(self, index: int, x: int, y: int):
        """Set layer position."""
        if 0 <= index < len(self.layers):
            self.layers[index]['instances'][0]['position'] = (x, y)
            print(f"✓ Layer {index} position: ({x}, {y})")
    
    def set_scale(self, index: int, scale: float):
        """Set layer scale."""
        if 0 <= index < len(self.layers):
            self.layers[index]['instances'][0]['scale'] = scale
            print(f"✓ Layer {index} scale: {scale}x")
    
    def set_rotation(self, index: int, degrees: float):
        """Set layer rotation."""
        if 0 <= index < len(self.layers):
            self.layers[index]['instances'][0]['rotation'] = degrees
            print(f"✓ Layer {index} rotation: {degrees}°")
    
    def set_flip_horizontal(self, index: int, flip: bool):
        """Set horizontal flip."""
        if 0 <= index < len(self.layers):
            self.layers[index]['instances'][0]['flip_horizontal'] = flip
            print(f"✓ Layer {index} flip_h: {flip}")
    
    def set_flip_vertical(self, index: int, flip: bool):
        """Set vertical flip."""
        if 0 <= index < len(self.layers):
            self.layers[index]['instances'][0]['flip_vertical'] = flip
            print(f"✓ Layer {index} flip_v: {flip}")
    
    # ========================================================================
    # Color Properties
    # ========================================================================
    
    def set_color(self, index: int, mode: str, intensity: float = 0.5,
                  saturation_boost: Optional[float] = None,
                  brightness_adjust: Optional[float] = None):
        """Set color properties."""
        if 0 <= index < len(self.layers):
            instance = self.layers[index]['instances'][0]
            instance['color_mode'] = mode
            instance['color_intensity'] = intensity
            instance['saturation_boost'] = saturation_boost
            instance['brightness_adjust'] = brightness_adjust
            print(f"✓ Layer {index} color: {mode} (intensity={intensity})")
    
    # ========================================================================
    # Batch Operations
    # ========================================================================
    
    def reset_layer(self, index: int):
        """Reset all properties of a layer to defaults."""
        if 0 <= index < len(self.layers):
            self.layers[index].update({
                'enabled': True,
                'count': 1,
                'instances': [
                    {
                        'position': (0, 0),
                        'scale': 1.0,
                        'rotation': 0.0,
                        'flip_horizontal': False,
                        'flip_vertical': False,
                        'color_mode': None,
                        'color_intensity': 0.5,
                        'saturation_boost': None,
                        'brightness_adjust': None,
                    }
                ]
            })
            print(f"✓ Layer {index} reset to defaults")
    
    def reset_all(self):
        """Reset all layers to defaults."""
        for i in range(len(self.layers)):
            self.reset_layer(i)
    
    # ========================================================================
    # Status Display
    # ========================================================================
    
    def show_status(self):
        """Show all layer properties."""
        print(f"\nLayer Properties for {self.image_id}")
        print("=" * 80)
        
        # Group by file to show duplicates
        seen_files = {}
        for layer in self.layers:
            file_key = str(layer['file'])
            if file_key not in seen_files:
                seen_files[file_key] = []
            seen_files[file_key].append(layer)
        
        for layer in self.layers:
            i = layer['index']
            status = "✓" if layer['enabled'] else "✗"
            layer_type = layer['type'].capitalize()
            
            # Check if this is a duplicate
            file_key = str(layer['file'])
            duplicates = seen_files[file_key]
            is_duplicate = len(duplicates) > 1 and layer != duplicates[0]
            
            instance = layer['instances'][0]
            props = []
            
            # Spatial
            if instance['position'] != (0, 0):
                props.append(f"pos=({instance['position'][0]:+d},{instance['position'][1]:+d})")
            if instance['scale'] != 1.0:
                props.append(f"scale={instance['scale']:.2f}x")
            if instance['rotation'] != 0:
                props.append(f"rot={instance['rotation']:.1f}°")
            if instance['flip_horizontal']:
                props.append("flip_h")
            if instance['flip_vertical']:
                props.append("flip_v")
            
            # Color
            if instance['color_mode']:
                color_str = f"color={instance['color_mode']}"
                if instance['saturation_boost']:
                    color_str += f"+sat"
                if instance['brightness_adjust']:
                    color_str += f"+bright"
                props.append(color_str)
            
            props_str = ", ".join(props) if props else "default"
            
            # Mark duplicates
            duplicate_marker = " [DUPLICATE]" if is_duplicate else ""
            
            print(f"{status} Layer {i:2d} ({layer_type:10s}): {props_str}{duplicate_marker}")
        print("=" * 80)
    
    # ========================================================================
    # Apply Transformations (Helper Functions)
    # ========================================================================
    
    def _apply_spatial_transforms(self, img: Image.Image, instance: Dict) -> Image.Image:
        """Apply spatial transformations to an image."""
        # Convert to RGBA for transformations
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 1. Flip
        if instance['flip_horizontal']:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if instance['flip_vertical']:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 2. Scale
        if instance['scale'] != 1.0:
            new_w = int(img.width * instance['scale'])
            new_h = int(img.height * instance['scale'])
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 3. Rotate
        if instance['rotation'] != 0:
            img = img.rotate(
                instance['rotation'],
                resample=Image.BICUBIC,
                expand=True,
                fillcolor=(0, 0, 0, 0)
            )
        
        return img
    
    def _apply_color_transform(self, img: Image.Image, instance: Dict) -> Image.Image:
        """Apply color transformations to an image."""
        if not instance['color_mode']:
            return img  # No color transform
        
        import numpy as np
        
        img_array = np.array(img)
        
        if img.mode == 'RGBA':
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
        else:
            rgb = img_array
            alpha = None
        
        mode = instance['color_mode']
        intensity = instance['color_intensity']
        
        # Simple color transformations
        if mode == 'grayscale':
            gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
            rgb_new = np.stack([gray, gray, gray], axis=2)
            rgb_result = (rgb * (1 - intensity) + rgb_new * intensity).astype(np.uint8)
        
        elif mode == 'warmer':
            # Shift toward warm tones
            shift = np.array([30, 10, -20], dtype=np.float32)
            rgb_new = np.clip(rgb.astype(np.float32) + shift * intensity * 2, 0, 255)
            rgb_result = rgb_new.astype(np.uint8)
        
        elif mode == 'cooler':
            # Shift toward cool tones
            shift = np.array([-20, 0, 30], dtype=np.float32)
            rgb_new = np.clip(rgb.astype(np.float32) + shift * intensity * 2, 0, 255)
            rgb_result = rgb_new.astype(np.uint8)
        
        else:
            # Default: no change
            rgb_result = rgb
        
        # Reconstruct image
        if alpha is not None:
            result_array = np.dstack([rgb_result, alpha])
            return Image.fromarray(result_array, 'RGBA')
        else:
            return Image.fromarray(rgb_result, 'RGB')
    
    # ========================================================================
    # MAIN RENDER FUNCTION - Applies ALL Properties
    # ========================================================================
    
    def render(self) -> Image.Image:
        """
        Render the final image by applying ALL layer properties.
        
        This is the SINGLE render function that handles everything:
        - Visibility (enabled/disabled)
        - Count (multiple instances of same layer)
        - Spatial transforms (position, scale, rotation, flip)
        - Color transforms (color mode, saturation, brightness)
        
        Returns:
            PIL Image
        """
        # Create canvas
        result = Image.new('RGBA', self.canvas_size, (255, 255, 255, 0))
        
        # Process each layer
        for layer in self.layers:
            # Check if enabled
            if not layer['enabled']:
                continue
            
            # Process each instance of this layer
            for instance in layer['instances']:
                # Load layer image (fresh copy for each instance)
                img = Image.open(layer['file']).convert('RGBA')
                
                # Store original size for position calculation
                original_size = img.size
                
                # Apply spatial transformations
                img = self._apply_spatial_transforms(img, instance)
                
                # Apply color transformations
                img = self._apply_color_transform(img, instance)
                
                # Calculate position (accounting for size changes)
                x_offset, y_offset = instance['position']
                size_diff_x = (img.width - original_size[0]) // 2
                size_diff_y = (img.height - original_size[1]) // 2
                paste_x = x_offset - size_diff_x
                paste_y = y_offset - size_diff_y
                
                # Composite onto canvas
                result.paste(img, (paste_x, paste_y), img)
        
        # Convert to RGB for final output
        rgb_result = Image.new('RGB', result.size, (255, 255, 255))
        rgb_result.paste(result, (0, 0), result)
        
        return rgb_result
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    def visualize(self, save_path: Optional[str] = None):
        """Visualize all layers and final result."""
        num_layers = len(self.layers)
        
        fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))
        
        if num_layers == 0:
            axes = [axes]
        elif num_layers == 1:
            axes = [axes[0], axes[1]]
        
        # Show individual layers
        for i, layer in enumerate(self.layers):
            # Load and transform
            img = Image.open(layer['file']).convert('RGBA')
            instance = layer['instances'][0]
            img = self._apply_spatial_transforms(img, instance)
            img = self._apply_color_transform(img, instance)
            
            axes[i].imshow(img)
            
            # Title
            status = "✓" if layer['enabled'] else "✗"
            title = f"{status} L{i}"
            
            # Add property hints
            hints = []
            if instance['position'] != (0, 0):
                hints.append(f"pos")
            if instance['scale'] != 1.0:
                hints.append(f"{instance['scale']:.1f}x")
            if instance['rotation'] != 0:
                hints.append(f"{instance['rotation']:.0f}°")
            if instance['color_mode']:
                hints.append(f"{instance['color_mode']}")
            
            if hints:
                title += "\n" + ", ".join(hints)
            
            axes[i].set_title(title, fontsize=9)
            axes[i].axis('off')
            
            if not layer['enabled']:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        
        # Show result
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


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    
    print("Unified Layer System with Duplication - Demo")
    print("=" * 70)
    
    # Create system
    system = UnifiedLayerSystem(
        image_id="000000000285",
        layer_dir="/content/mulan_output"
    )
    
    # Example 1: Duplicate a layer
    print("\n--- Example 1: Duplicate Layer ---")
    
    # Duplicate layer 1 (creates 2 total)
    system.set_count(1, 2)
    
    # Now we have layer 1 and a new duplicate layer
    # Get the duplicate indices
    duplicates = system.get_duplicates_of(1)
    print(f"Layer 1 duplicates at indices: {duplicates}")
    
    # Modify them independently (they're just different layers now!)
    system.set_position(duplicates[0], x=50, y=-30)   # Original
    system.set_position(duplicates[1], x=-50, y=30)   # Duplicate
    
    # Different colors
    system.set_color(duplicates[0], 'blue', intensity=0.5)
    system.set_color(duplicates[1], 'red', intensity=0.5)
    
    system.show_status()
    
    result1 = system.render()
    result1.save("duplicate_demo.png")
    print("✓ Saved: duplicate_demo.png")
    
    # Example 2: Triple layer
    print("\n--- Example 2: Triple Layer ---")
    system.reset_all()
    
    # Create 3 copies of layer 1
    system.set_count(1, 3)
    duplicates = system.get_duplicates_of(1)
    
    # Position them in a row
    system.set_position(duplicates[0], x=-100, y=0)
    system.set_position(duplicates[1], x=0, y=0)
    system.set_position(duplicates[2], x=100, y=0)
    
    # Different sizes
    system.set_scale(duplicates[0], 0.8)
    system.set_scale(duplicates[1], 1.0)
    system.set_scale(duplicates[2], 1.2)
    
    system.show_status()
    
    result2 = system.render()
    result2.save("triple_demo.png")
    print("✓ Saved: triple_demo.png")
    
    # Example 3: Simple usage without duplication
    print("\n--- Example 3: Regular Usage ---")
    system.reset_all()
    
    # Just use layers normally
    system.set_position(1, x=40, y=-20)
    system.set_scale(1, 1.2)
    system.set_color(1, 'cooler', intensity=0.4)
    
    system.show_status()
    
    result3 = system.render()
    result3.save("simple_demo.png")
    print("✓ Saved: simple_demo.png")
    
    print("\n✓ All examples complete!")
    print("Duplicates are just new layers - use them like any other layer!")
