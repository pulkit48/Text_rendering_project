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
    # Spatial Properties
    # ========================================================================
    
    def set_position(self, index: int, x: int, y: int):
        """Set layer position."""
        if 0 <= index < len(self.layers):
            self.layers[index]['position'] = (x, y)
            print(f"✓ Layer {index} position: ({x}, {y})")
    
    def set_scale(self, index: int, scale: float):
        """Set layer scale."""
        if 0 <= index < len(self.layers):
            self.layers[index]['scale'] = scale
            print(f"✓ Layer {index} scale: {scale}x")
    
    def set_rotation(self, index: int, degrees: float):
        """Set layer rotation."""
        if 0 <= index < len(self.layers):
            self.layers[index]['rotation'] = degrees
            print(f"✓ Layer {index} rotation: {degrees}°")
    
    def set_flip_horizontal(self, index: int, flip: bool):
        """Set horizontal flip."""
        if 0 <= index < len(self.layers):
            self.layers[index]['flip_horizontal'] = flip
            print(f"✓ Layer {index} flip_h: {flip}")
    
    def set_flip_vertical(self, index: int, flip: bool):
        """Set vertical flip."""
        if 0 <= index < len(self.layers):
            self.layers[index]['flip_vertical'] = flip
            print(f"✓ Layer {index} flip_v: {flip}")
    
    # ========================================================================
    # Color Properties
    # ========================================================================
    
    def set_color(self, index: int, mode: str, intensity: float = 0.5,
                  saturation_boost: Optional[float] = None,
                  brightness_adjust: Optional[float] = None):
        """Set color properties."""
        if 0 <= index < len(self.layers):
            self.layers[index]['color_mode'] = mode
            self.layers[index]['color_intensity'] = intensity
            self.layers[index]['saturation_boost'] = saturation_boost
            self.layers[index]['brightness_adjust'] = brightness_adjust
            print(f"✓ Layer {index} color: {mode} (intensity={intensity})")
    
    # ========================================================================
    # Batch Operations
    # ========================================================================
    
    def reset_layer(self, index: int):
        """Reset all properties of a layer to defaults."""
        if 0 <= index < len(self.layers):
            self.layers[index].update({
                'enabled': True,
                'position': (0, 0),
                'scale': 1.0,
                'rotation': 0.0,
                'flip_horizontal': False,
                'flip_vertical': False,
                'color_mode': None,
                'color_intensity': 0.5,
                'saturation_boost': None,
                'brightness_adjust': None,
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
        for layer in self.layers:
            i = layer['index']
            status = "✓" if layer['enabled'] else "✗"
            layer_type = layer['type'].capitalize()
            
            # Build property list
            props = []
            
            # Spatial
            if layer['position'] != (0, 0):
                props.append(f"pos=({layer['position'][0]:+d},{layer['position'][1]:+d})")
            if layer['scale'] != 1.0:
                props.append(f"scale={layer['scale']:.2f}x")
            if layer['rotation'] != 0:
                props.append(f"rot={layer['rotation']:.1f}°")
            if layer['flip_horizontal']:
                props.append("flip_h")
            if layer['flip_vertical']:
                props.append("flip_v")
            
            # Color
            if layer['color_mode']:
                color_str = f"color={layer['color_mode']}"
                if layer['saturation_boost']:
                    color_str += f"+sat"
                if layer['brightness_adjust']:
                    color_str += f"+bright"
                props.append(color_str)
            
            props_str = ", ".join(props) if props else "default"
            print(f"{status} Layer {i:2d} ({layer_type:10s}): {props_str}")
        print("=" * 80)
    
    # ========================================================================
    # Apply Transformations (Helper Functions)
    # ========================================================================
    
    def _apply_spatial_transforms(self, img: Image.Image, layer: Dict) -> Image.Image:
        """Apply spatial transformations to an image."""
        # Convert to RGBA for transformations
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 1. Flip
        if layer['flip_horizontal']:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if layer['flip_vertical']:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 2. Scale
        if layer['scale'] != 1.0:
            new_w = int(img.width * layer['scale'])
            new_h = int(img.height * layer['scale'])
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 3. Rotate
        if layer['rotation'] != 0:
            img = img.rotate(
                layer['rotation'],
                resample=Image.BICUBIC,
                expand=True,
                fillcolor=(0, 0, 0, 0)
            )
        
        return img
    
    def _apply_color_transform(self, img: Image.Image, layer: Dict) -> Image.Image:
        """Apply color transformations to an image."""
        if not layer['color_mode']:
            return img  # No color transform
        
        import numpy as np
        
        img_array = np.array(img)
        
        if img.mode == 'RGBA':
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
        else:
            rgb = img_array
            alpha = None
        
        mode = layer['color_mode']
        intensity = layer['color_intensity']
        
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
            
            # Load layer image
            img = Image.open(layer['file']).convert('RGBA')
            
            # Store original size for position calculation
            original_size = img.size
            
            # Apply spatial transformations
            img = self._apply_spatial_transforms(img, layer)
            
            # Apply color transformations
            img = self._apply_color_transform(img, layer)
            
            # Calculate position (accounting for size changes)
            x_offset, y_offset = layer['position']
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
        
        # Show individual layers
        for i, layer in enumerate(self.layers):
            # Load and transform
            img = Image.open(layer['file']).convert('RGBA')
            img = self._apply_spatial_transforms(img, layer)
            img = self._apply_color_transform(img, layer)
            
            axes[i].imshow(img)
            
            # Title
            status = "✓" if layer['enabled'] else "✗"
            title = f"{status} L{i}"
            
            # Add property hints
            hints = []
            if layer['position'] != (0, 0):
                hints.append(f"pos")
            if layer['scale'] != 1.0:
                hints.append(f"{layer['scale']:.1f}x")
            if layer['rotation'] != 0:
                hints.append(f"{layer['rotation']:.0f}°")
            if layer['color_mode']:
                hints.append(f"{layer['color_mode']}")
            
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
    
    print("Unified Layer System - Demo")
    print("=" * 70)
    
    # Create system
    system = UnifiedLayerSystem(
        image_id="000000000285",
        layer_dir="/content/mulan_output"
    )
    
    # Example 1: Modify different properties
    print("\n--- Setting Properties ---")
    
    # Layer 1: Position + Scale
    system.set_position(1, x=50, y=-30)
    system.set_scale(1, 1.3)
    
    # Layer 2: Color + Rotation
    system.set_color(2, 'warmer', intensity=0.5)
    system.set_rotation(2, 15)
    
    # Layer 3: Flip
    system.set_flip_horizontal(3, True)
    
    # Show status
    system.show_status()
    
    # Render (applies ALL properties)
    print("\n--- Rendering ---")
    result = system.render()
    result.save("unified_demo.png")
    print("✓ Saved: unified_demo.png")
    
    # Example 2: Combine everything
    print("\n--- Complex Example ---")
    system.reset_all()
    
    # Layer 1: Everything!
    system.set_position(1, x=40, y=-20)
    system.set_scale(1, 1.2)
    system.set_rotation(1, 10)
    system.set_color(1, 'cooler', intensity=0.4)
    
    system.show_status()
    
    result2 = system.render()
    result2.save("unified_complex.png")
    print("✓ Saved: unified_complex.png")
    
    # Visualize
    print("\n--- Visualization ---")
    system.visualize(save_path="unified_visualization.png")
    
    print("\n✓ All examples complete!")
    print("One render() function applies all properties!")
