"""
Layer Spatial Transformation Tool

Handle position, scale, rotation, and other spatial manipulations of layers.
"""

from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class LayerTransformer:
    """
    Tool to handle spatial transformations of layers.
    
    Features:
    - Move layers (x, y position)
    - Scale layers (resize)
    - Rotate layers
    - Flip layers
    - Visualize transformations
    """
    
    def __init__(self, image_id: str, layer_dir: str):
        """
        Initialize the transformer.
        
        Args:
            image_id: Image identifier (e.g., "000000000285")
            layer_dir: Directory containing layer files
        """
        self.image_id = image_id
        self.layer_dir = Path(layer_dir)
        
        # Find all layer files
        self.layer_files = sorted(self.layer_dir.glob(f"{image_id}-layer_*.png"))
        
        # Track transformations for each layer
        self.transformations = {}
        for i in range(len(self.layer_files)):
            self.transformations[i] = {
                'position': (0, 0),      # (x, y) offset in pixels
                'scale': 1.0,            # Scale factor (1.0 = original size)
                'rotation': 0,           # Rotation in degrees
                'flip_horizontal': False,
                'flip_vertical': False,
                'enabled': True
            }
        
        # Get canvas size from background
        bg = Image.open(self.layer_files[0])
        self.canvas_size = bg.size
        
        print(f"Loaded {len(self.layer_files)} layers for {image_id}")
        print(f"Canvas size: {self.canvas_size[0]}x{self.canvas_size[1]}")
    
    def get_num_layers(self):
        """Get total number of layers."""
        return len(self.layer_files)
    
    # ========================================================================
    # Position Control
    # ========================================================================
    
    def set_position(self, layer_index: int, x: int, y: int):
        """
        Set layer position offset.
        
        Args:
            layer_index: Which layer
            x: Horizontal offset (positive = right, negative = left)
            y: Vertical offset (positive = down, negative = up)
        """
        if 0 <= layer_index < len(self.layer_files):
            self.transformations[layer_index]['position'] = (x, y)
            print(f"✓ Layer {layer_index} position: ({x}, {y})")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    def move_layer(self, layer_index: int, dx: int, dy: int):
        """
        Move layer relative to current position.
        
        Args:
            layer_index: Which layer
            dx: Horizontal movement
            dy: Vertical movement
        """
        if 0 <= layer_index < len(self.layer_files):
            current_x, current_y = self.transformations[layer_index]['position']
            new_x = current_x + dx
            new_y = current_y + dy
            self.transformations[layer_index]['position'] = (new_x, new_y)
            print(f"✓ Layer {layer_index} moved by ({dx}, {dy}) → now at ({new_x}, {new_y})")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    def center_layer(self, layer_index: int):
        """Center a layer on the canvas."""
        if 0 <= layer_index < len(self.layer_files):
            # This will center after scaling, handled in render
            self.transformations[layer_index]['position'] = (0, 0)
            print(f"✓ Layer {layer_index} will be centered")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    # ========================================================================
    # Scale Control
    # ========================================================================
    
    def set_scale(self, layer_index: int, scale: float):
        """
        Set layer scale.
        
        Args:
            layer_index: Which layer
            scale: Scale factor (0.5 = half size, 2.0 = double size)
        """
        if 0 <= layer_index < len(self.layer_files):
            if scale <= 0:
                print(f"✗ Scale must be positive, got {scale}")
                return
            self.transformations[layer_index]['scale'] = scale
            print(f"✓ Layer {layer_index} scale: {scale}x")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    def resize_layer(self, layer_index: int, width: int, height: int):
        """
        Resize layer to specific dimensions.
        
        Args:
            layer_index: Which layer
            width: New width in pixels
            height: New height in pixels
        """
        if 0 <= layer_index < len(self.layer_files):
            # Load original to get current size
            img = Image.open(self.layer_files[layer_index])
            scale_x = width / img.width
            scale_y = height / img.height
            # Use average scale to maintain aspect ratio
            scale = (scale_x + scale_y) / 2
            self.transformations[layer_index]['scale'] = scale
            print(f"✓ Layer {layer_index} resized to ~{width}x{height} (scale: {scale:.2f}x)")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    # ========================================================================
    # Rotation Control
    # ========================================================================
    
    def set_rotation(self, layer_index: int, degrees: float):
        """
        Set layer rotation.
        
        Args:
            layer_index: Which layer
            degrees: Rotation angle (positive = counter-clockwise)
        """
        if 0 <= layer_index < len(self.layer_files):
            self.transformations[layer_index]['rotation'] = degrees
            print(f"✓ Layer {layer_index} rotation: {degrees}°")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    def rotate_layer(self, layer_index: int, degrees: float):
        """
        Rotate layer relative to current rotation.
        
        Args:
            layer_index: Which layer
            degrees: Additional rotation
        """
        if 0 <= layer_index < len(self.layer_files):
            current = self.transformations[layer_index]['rotation']
            new_rotation = (current + degrees) % 360
            self.transformations[layer_index]['rotation'] = new_rotation
            print(f"✓ Layer {layer_index} rotated by {degrees}° → now at {new_rotation}°")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    # ========================================================================
    # Flip Control
    # ========================================================================
    
    def flip_horizontal(self, layer_index: int, flip: bool = True):
        """Flip layer horizontally (mirror left-right)."""
        if 0 <= layer_index < len(self.layer_files):
            self.transformations[layer_index]['flip_horizontal'] = flip
            status = "enabled" if flip else "disabled"
            print(f"✓ Layer {layer_index} horizontal flip: {status}")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    def flip_vertical(self, layer_index: int, flip: bool = True):
        """Flip layer vertically (mirror top-bottom)."""
        if 0 <= layer_index < len(self.layer_files):
            self.transformations[layer_index]['flip_vertical'] = flip
            status = "enabled" if flip else "disabled"
            print(f"✓ Layer {layer_index} vertical flip: {status}")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    # ========================================================================
    # Enable/Disable
    # ========================================================================
    
    def enable_layer(self, layer_index: int):
        """Enable a layer."""
        if 0 <= layer_index < len(self.layer_files):
            self.transformations[layer_index]['enabled'] = True
            print(f"✓ Layer {layer_index} enabled")
    
    def disable_layer(self, layer_index: int):
        """Disable a layer."""
        if 0 <= layer_index < len(self.layer_files):
            self.transformations[layer_index]['enabled'] = False
            print(f"✓ Layer {layer_index} disabled")
    
    # ========================================================================
    # Reset
    # ========================================================================
    
    def reset_layer(self, layer_index: int):
        """Reset all transformations for a layer."""
        if 0 <= layer_index < len(self.layer_files):
            self.transformations[layer_index] = {
                'position': (0, 0),
                'scale': 1.0,
                'rotation': 0,
                'flip_horizontal': False,
                'flip_vertical': False,
                'enabled': True
            }
            print(f"✓ Layer {layer_index} reset to defaults")
    
    def reset_all(self):
        """Reset all transformations for all layers."""
        for i in range(len(self.layer_files)):
            self.reset_layer(i)
        print("✓ All layers reset")
    
    # ========================================================================
    # Status
    # ========================================================================
    
    def show_status(self):
        """Print current transformation status."""
        print(f"\nTransformation Status for {self.image_id}")
        print("=" * 70)
        for i in range(len(self.layer_files)):
            trans = self.transformations[i]
            layer_type = "Background" if i == 0 else f"Instance  "
            
            status = "✓" if trans['enabled'] else "✗"
            pos = trans['position']
            scale = trans['scale']
            rot = trans['rotation']
            
            info = f"{status} Layer {i} ({layer_type}): "
            
            changes = []
            if pos != (0, 0):
                changes.append(f"pos=({pos[0]:+d}, {pos[1]:+d})")
            if scale != 1.0:
                changes.append(f"scale={scale:.2f}x")
            if rot != 0:
                changes.append(f"rot={rot:.1f}°")
            if trans['flip_horizontal']:
                changes.append("flip_h")
            if trans['flip_vertical']:
                changes.append("flip_v")
            
            if changes:
                info += ", ".join(changes)
            else:
                info += "default"
            
            print(info)
        print("=" * 70)
    
    # ========================================================================
    # Apply Transformations
    # ========================================================================
    
    def _apply_transformations(self, img: Image.Image, layer_index: int) -> Image.Image:
        """Apply all transformations to a layer."""
        trans = self.transformations[layer_index]
        
        # 1. Flip
        if trans['flip_horizontal']:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if trans['flip_vertical']:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 2. Scale
        if trans['scale'] != 1.0:
            new_width = int(img.width * trans['scale'])
            new_height = int(img.height * trans['scale'])
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 3. Rotate
        if trans['rotation'] != 0:
            img = img.rotate(trans['rotation'], 
                           resample=Image.BICUBIC, 
                           expand=True)
        
        return img
    
    # ========================================================================
    # Render
    # ========================================================================
    
    def render(self) -> Image.Image:
        """
        Render the final composite image with all transformations.
        
        Returns:
            PIL Image
        """
        # Create canvas
        result = Image.new('RGBA', self.canvas_size, (255, 255, 255, 255))
        
        # Process each layer
        for i in range(len(self.layer_files)):
            trans = self.transformations[i]
            
            if not trans['enabled']:
                continue
            
            # Load layer
            if i == 0:
                # Background
                img = Image.open(self.layer_files[i]).convert('RGB')
            else:
                # Instance
                img = Image.open(self.layer_files[i]).convert('RGBA')
            
            # Apply transformations
            img = self._apply_transformations(img, i)
            
            # Calculate paste position
            x_offset, y_offset = trans['position']
            
            # Center if needed
            paste_x = x_offset
            paste_y = y_offset
            
            # Paste layer
            if i == 0:
                # Background - paste as RGB
                result.paste(img, (paste_x, paste_y))
            else:
                # Instance - use alpha
                result.paste(img, (paste_x, paste_y), img if img.mode == 'RGBA' else None)
        
        # Convert to RGB for saving
        if result.mode == 'RGBA':
            rgb_result = Image.new('RGB', result.size, (255, 255, 255))
            rgb_result.paste(result, (0, 0), result)
            return rgb_result
        
        return result
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    def visualize(self, save_path: Optional[str] = None):
        """
        Visualize all layers and the combined result with transformation info.
        
        Args:
            save_path: Optional path to save visualization
        """
        num_layers = len(self.layer_files)
        
        # Create figure
        fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))
        
        if num_layers == 0:
            axes = [axes]
        
        # Show individual layers
        for i, layer_file in enumerate(self.layer_files):
            trans = self.transformations[i]
            
            # Load and transform
            if i == 0:
                img = Image.open(layer_file).convert('RGB')
            else:
                img = Image.open(layer_file).convert('RGBA')
            
            img_transformed = self._apply_transformations(img, i)
            
            axes[i].imshow(img_transformed)
            
            # Title with transformation info
            layer_type = "BG" if i == 0 else f"I{i}"
            status = "✓" if trans['enabled'] else "✗"
            
            title_parts = [f"{status} L{i} ({layer_type})"]
            
            if trans['position'] != (0, 0):
                title_parts.append(f"pos({trans['position'][0]:+d},{trans['position'][1]:+d})")
            if trans['scale'] != 1.0:
                title_parts.append(f"{trans['scale']:.1f}x")
            if trans['rotation'] != 0:
                title_parts.append(f"{trans['rotation']:.0f}°")
            
            axes[i].set_title("\n".join(title_parts), fontsize=9)
            axes[i].axis('off')
            
            # Red border for disabled
            if not trans['enabled']:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        
        # Show combined result
        combined = self.render()
        axes[-1].imshow(combined)
        axes[-1].set_title("Combined\nResult", fontsize=11, fontweight='bold')
        axes[-1].axis('off')
        
        # Green border
        for spine in axes[-1].spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization: {save_path}")
        
        plt.show()
        
        return fig


# ============================================================================
# Demo Usage
# ============================================================================

if __name__ == "__main__":
    
    print("Layer Spatial Transformer - Demo")
    print("=" * 60)
    
    # Create transformer
    transformer = LayerTransformer(
        image_id="000000000285",
        layer_dir="/content/mulan_output"
    )
    
    # Example 1: Move layers
    print("\n--- Example 1: Move Layers ---")
    transformer.set_position(1, x=50, y=-30)   # Move instance 1 right and up
    transformer.set_position(2, x=-40, y=20)   # Move instance 2 left and down
    
    transformer.show_status()
    result1 = transformer.render()
    result1.save("transform_1_moved.png")
    print("✓ Saved: transform_1_moved.png")
    
    # Example 2: Scale layers
    print("\n--- Example 2: Scale Layers ---")
    transformer.reset_all()
    transformer.set_scale(1, 1.5)   # Make instance 1 bigger
    transformer.set_scale(2, 0.7)   # Make instance 2 smaller
    
    result2 = transformer.render()
    result2.save("transform_2_scaled.png")
    print("✓ Saved: transform_2_scaled.png")
    
    # Example 3: Rotate layers
    print("\n--- Example 3: Rotate Layers ---")
    transformer.reset_all()
    transformer.set_rotation(1, 15)    # Rotate 15° counter-clockwise
    transformer.set_rotation(2, -30)   # Rotate 30° clockwise
    
    result3 = transformer.render()
    result3.save("transform_3_rotated.png")
    print("✓ Saved: transform_3_rotated.png")
    
    # Example 4: Flip layers
    print("\n--- Example 4: Flip Layers ---")
    transformer.reset_all()
    transformer.flip_horizontal(1)  # Mirror instance 1
    
    result4 = transformer.render()
    result4.save("transform_4_flipped.png")
    print("✓ Saved: transform_4_flipped.png")
    
    # Example 5: Combine multiple transformations
    print("\n--- Example 5: Combined Transformations ---")
    transformer.reset_all()
    
    # Instance 1: move, scale, rotate
    transformer.set_position(1, x=30, y=-20)
    transformer.set_scale(1, 1.3)
    transformer.set_rotation(1, 10)
    
    # Instance 2: move, scale, flip
    transformer.set_position(2, x=-50, y=30)
    transformer.set_scale(2, 0.8)
    transformer.flip_horizontal(2)
    
    transformer.show_status()
    result5 = transformer.render()
    result5.save("transform_5_combined.png")
    print("✓ Saved: transform_5_combined.png")
    
    # Example 6: Visualize
    print("\n--- Example 6: Visualization ---")
    transformer.visualize(save_path="transform_visualization.png")
    
    print("\n✓ All examples complete!")
    print("Created 6 images showing different transformations.")
