"""
Layer Combination Tool

Features:
1. Combine layers
2. Enable/disable specific layers
3. Visualize all layers + combined result
"""

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


class LayerCombiner:
    """Simple tool to combine image layers with control."""
    
    def __init__(self, image_id, layer_dir):
        """
        Initialize the tool.
        
        Args:
            image_id: e.g., "000000000285"
            layer_dir: Directory containing layer files
        """
        self.image_id = image_id
        self.layer_dir = Path(layer_dir)
        
        # Find all layer files
        self.layer_files = sorted(self.layer_dir.glob(f"{image_id}-layer_*.png"))
        
        # Track which layers are enabled (all enabled by default)
        self.enabled = [True] * len(self.layer_files)
        
        # Track color alterations for each layer
        self.layer_colors = {}  # layer_index -> color_mode
        
        print(f"Loaded {len(self.layer_files)} layers for image {image_id}")
    
    def get_num_layers(self):
        """Get total number of layers."""
        return len(self.layer_files)
    
    def enable_layer(self, layer_index):
        """Enable a specific layer."""
        if 0 <= layer_index < len(self.enabled):
            self.enabled[layer_index] = True
            print(f"✓ Enabled layer {layer_index}")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    def disable_layer(self, layer_index):
        """Disable a specific layer."""
        if 0 <= layer_index < len(self.enabled):
            self.enabled[layer_index] = False
            print(f"✓ Disabled layer {layer_index}")
        else:
            print(f"✗ Invalid layer index: {layer_index}")
    
    def enable_all(self):
        """Enable all layers."""
        self.enabled = [True] * len(self.layer_files)
        print("✓ Enabled all layers")
    
    def disable_all(self):
        """Disable all layers."""
        self.enabled = [False] * len(self.layer_files)
        print("✓ Disabled all layers")
    
    def set_layer_color(self, layer_index, color_mode, intensity=0.5, 
                       saturation_boost=None, brightness_adjust=None):
        """
        Change the color of a layer NATURALLY while preserving texture.
        
        This uses HSV color space for natural-looking transformations.
        
        Args:
            layer_index: Which layer to modify
            color_mode: How to change color:
                       
                       Simple modes:
                       - 'warmer' / 'cooler' (temperature shift)
                       - 'more_saturated' / 'less_saturated'
                       - 'brighter' / 'darker'
                       - 'sepia', 'grayscale'
                       
                       Hue shift modes (natural color changes):
                       - 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink'
                       
                       Direct hue value:
                       - Hue number 0-360 (e.g., 120 for green)
                       
            intensity: How strong the effect (0.0 to 1.0)
            saturation_boost: Optional - increase/decrease saturation (-1.0 to 1.0)
            brightness_adjust: Optional - brighten/darken (-1.0 to 1.0)
        
        Examples:
            # Make a brown dog more golden/orange
            tool.set_layer_color(1, 'warmer', intensity=0.3)
            
            # Make grass more vibrant green
            tool.set_layer_color(2, 'green', intensity=0.4, saturation_boost=0.2)
            
            # Shift to blue tones (like underwater effect)
            tool.set_layer_color(0, 'cyan', intensity=0.5)
            
            # Make object less saturated (more grayish but keep hue)
            tool.set_layer_color(1, 'less_saturated', intensity=0.4)
        """
        if not 0 <= layer_index < len(self.layer_files):
            print(f"✗ Invalid layer index: {layer_index}")
            return
        
        if not 0.0 <= intensity <= 1.0:
            print(f"✗ Intensity must be between 0.0 and 1.0")
            return
        
        self.layer_colors[layer_index] = {
            'mode': color_mode,
            'intensity': intensity,
            'saturation_boost': saturation_boost,
            'brightness_adjust': brightness_adjust
        }
        
        desc = f"{color_mode} (intensity: {intensity}"
        if saturation_boost is not None:
            desc += f", saturation: {saturation_boost:+.1f}"
        if brightness_adjust is not None:
            desc += f", brightness: {brightness_adjust:+.1f}"
        desc += ")"
        
        print(f"✓ Set color for layer {layer_index}: {desc}")
    
    def reset_layer_color(self, layer_index):
        """Remove color alteration from a layer."""
        if layer_index in self.layer_colors:
            del self.layer_colors[layer_index]
            print(f"✓ Reset color for layer {layer_index}")
    
    def reset_all_colors(self):
        """Remove all color alterations."""
        self.layer_colors = {}
        print("✓ Reset all colors")
    
    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space (numpy arrays)."""
        import numpy as np
        
        rgb = rgb.astype(np.float32) / 255.0
        
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        diff = maxc - minc
        
        # Hue calculation
        h = np.zeros_like(maxc)
        
        mask = diff != 0
        r_mask = (maxc == r) & mask
        g_mask = (maxc == g) & mask
        b_mask = (maxc == b) & mask
        
        h[r_mask] = (60 * ((g[r_mask] - b[r_mask]) / diff[r_mask]) + 360) % 360
        h[g_mask] = (60 * ((b[g_mask] - r[g_mask]) / diff[g_mask]) + 120) % 360
        h[b_mask] = (60 * ((r[b_mask] - g[b_mask]) / diff[b_mask]) + 240) % 360
        
        # Saturation
        s = np.zeros_like(maxc)
        s[maxc != 0] = diff[maxc != 0] / maxc[maxc != 0]
        
        # Value (brightness)
        v = maxc
        
        return h, s, v
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB color space (numpy arrays)."""
        import numpy as np
        
        h = h % 360
        
        c = v * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = v - c
        
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        
        mask0 = (h >= 0) & (h < 60)
        mask1 = (h >= 60) & (h < 120)
        mask2 = (h >= 120) & (h < 180)
        mask3 = (h >= 180) & (h < 240)
        mask4 = (h >= 240) & (h < 300)
        mask5 = (h >= 300) & (h < 360)
        
        r[mask0], g[mask0], b[mask0] = c[mask0], x[mask0], 0
        r[mask1], g[mask1], b[mask1] = x[mask1], c[mask1], 0
        r[mask2], g[mask2], b[mask2] = 0, c[mask2], x[mask2]
        r[mask3], g[mask3], b[mask3] = 0, x[mask3], c[mask3]
        r[mask4], g[mask4], b[mask4] = x[mask4], 0, c[mask4]
        r[mask5], g[mask5], b[mask5] = c[mask5], 0, x[mask5]
        
        r = (r + m) * 255
        g = (g + m) * 255
        b = (b + m) * 255
        
        rgb = np.stack([r, g, b], axis=2)
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def _apply_color_to_layer(self, img, color_mode, intensity, saturation_boost=None, brightness_adjust=None):
        """
        Apply NATURAL color transformation using HSV color space.
        
        This preserves texture by working in HSV where:
        - H (Hue) = the color itself
        - S (Saturation) = how vivid/gray it is
        - V (Value) = brightness (this is the texture!)
        """
        import numpy as np
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Separate RGB and Alpha (if RGBA)
        if img.mode == 'RGBA':
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
        else:
            rgb = img_array
            alpha = None
        
        # Special cases that don't use HSV
        if color_mode == 'grayscale':
            gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
            rgb_new = np.stack([gray, gray, gray], axis=2).astype(np.uint8)
            rgb_result = (rgb * (1 - intensity) + rgb_new * intensity).astype(np.uint8)
            
        elif color_mode == 'sepia':
            r = rgb[:,:,0] * 0.393 + rgb[:,:,1] * 0.769 + rgb[:,:,2] * 0.189
            g = rgb[:,:,0] * 0.349 + rgb[:,:,1] * 0.686 + rgb[:,:,2] * 0.168
            b = rgb[:,:,0] * 0.272 + rgb[:,:,1] * 0.534 + rgb[:,:,2] * 0.131
            rgb_new = np.stack([r, g, b], axis=2)
            rgb_result = (rgb * (1 - intensity) + rgb_new * intensity).astype(np.uint8)
            
        else:
            # Convert to HSV for natural color manipulation
            h, s, v = self._rgb_to_hsv(rgb)
            
            # Clone for modification
            h_new = h.copy()
            s_new = s.copy()
            v_new = v.copy()
            
            # Apply color mode
            if color_mode == 'warmer':
                # Shift hues toward warm (red-orange-yellow)
                # Reds get more orange, greens get more yellow, blues get less blue
                h_new = h + 15 * intensity
                s_new = s * (1 + 0.1 * intensity)  # Slight saturation boost
                
            elif color_mode == 'cooler':
                # Shift toward cool (blue-cyan-green)
                h_new = h - 15 * intensity
                s_new = s * (1 + 0.1 * intensity)
                
            elif color_mode == 'more_saturated':
                # Increase color vividness
                s_new = s + (1 - s) * intensity
                
            elif color_mode == 'less_saturated':
                # Decrease saturation (more grayish)
                s_new = s * (1 - intensity)
                
            elif color_mode == 'brighter':
                # Increase brightness
                v_new = v + (1 - v) * intensity * 0.5
                
            elif color_mode == 'darker':
                # Decrease brightness
                v_new = v * (1 - intensity * 0.5)
            
            elif isinstance(color_mode, (int, float)):
                # Direct hue value (0-360)
                target_hue = float(color_mode)
                # Shift current hue toward target
                hue_diff = (target_hue - h + 180) % 360 - 180
                h_new = h + hue_diff * intensity
                
            else:
                # Named color modes - shift hue toward that color
                color_hues = {
                    'red': 0,
                    'orange': 30,
                    'yellow': 60,
                    'green': 120,
                    'cyan': 180,
                    'blue': 240,
                    'purple': 280,
                    'pink': 330
                }
                
                if color_mode in color_hues:
                    target_hue = color_hues[color_mode]
                    # Shift toward target hue
                    hue_diff = (target_hue - h + 180) % 360 - 180
                    h_new = h + hue_diff * intensity
                    # Boost saturation slightly for named colors
                    s_new = s * (1 + 0.2 * intensity)
                else:
                    print(f"⚠️  Unknown color mode: {color_mode}")
            
            # Apply optional adjustments
            if saturation_boost is not None:
                if saturation_boost > 0:
                    s_new = s_new + (1 - s_new) * saturation_boost
                else:
                    s_new = s_new * (1 + saturation_boost)
            
            if brightness_adjust is not None:
                if brightness_adjust > 0:
                    v_new = v_new + (1 - v_new) * brightness_adjust
                else:
                    v_new = v_new * (1 + brightness_adjust)
            
            # Ensure valid ranges
            s_new = np.clip(s_new, 0, 1)
            v_new = np.clip(v_new, 0, 1)
            
            # Convert back to RGB
            rgb_result = self._hsv_to_rgb(h_new, s_new, v_new)
        
        # Reconstruct image
        if alpha is not None:
            result_array = np.dstack([rgb_result, alpha])
            result_img = Image.fromarray(result_array, 'RGBA')
        else:
            result_img = Image.fromarray(rgb_result, 'RGB')
        
        return result_img
    
    def show_status(self):
        """Show which layers are enabled/disabled."""
        print(f"\nLayer Status for {self.image_id}:")
        print("-" * 50)
        for i, layer_file in enumerate(self.layer_files):
            status = "✓ ENABLED " if self.enabled[i] else "✗ DISABLED"
            layer_type = "Background" if i == 0 else f"Instance  "
            color_info = ""
            if hasattr(self, 'layer_colors') and i in self.layer_colors:
                color_info = f" [Color: {self.layer_colors[i]}]"
            print(f"  Layer {i} ({layer_type}): {status} - {layer_file.name}{color_info}")
        print("-" * 50)
    
    def combine(self):
        """
        Combine only the ENABLED layers with any color alterations applied.
        
        Returns:
            PIL Image
        """
        result = None
        
        # Process each layer in order
        for i, layer_file in enumerate(self.layer_files):
            
            if not self.enabled[i]:
                continue  # Skip disabled layers
            
            # Load the layer
            if i == 0:
                # Background layer
                img = Image.open(layer_file).convert('RGB')
                
                # Apply color alteration if set
                if i in self.layer_colors:
                    color_info = self.layer_colors[i]
                    img = self._apply_color_to_layer(
                        img, 
                        color_info['mode'], 
                        color_info['intensity'],
                        color_info.get('saturation_boost'),
                        color_info.get('brightness_adjust')
                    )
                
                result = img.copy()
            else:
                # Instance layer
                if result is None:
                    # No background, start with first enabled instance
                    img = Image.open(layer_file).convert('RGBA')
                    # Create blank canvas
                    result = Image.new('RGB', img.size, (255, 255, 255))
                
                img = Image.open(layer_file).convert('RGBA')
                
                # Apply color alteration if set
                if i in self.layer_colors:
                    color_info = self.layer_colors[i]
                    img = self._apply_color_to_layer(
                        img, 
                        color_info['mode'], 
                        color_info['intensity'],
                        color_info.get('saturation_boost'),
                        color_info.get('brightness_adjust')
                    )
                
                result.paste(img, (0, 0), img)
        
        if result is None:
            print("⚠️  Warning: No layers enabled!")
            # Return blank image
            sample = Image.open(self.layer_files[0])
            result = Image.new('RGB', sample.size, (255, 255, 255))
        
        return result
    
    def visualize(self, figsize=(20, 5), save_path=None):
        """
        Visualize all layers and the combined result.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the visualization
        """
        num_layers = len(self.layer_files)
        
        # Create subplots: all layers + combined result
        fig, axes = plt.subplots(1, num_layers + 1, figsize=figsize)
        
        # If only one subplot, make it a list
        if num_layers == 0:
            axes = [axes]
        
        # Show individual layers
        for i, layer_file in enumerate(self.layer_files):
            img = Image.open(layer_file)
            axes[i].imshow(img)
            
            # Title
            layer_type = "Background" if i == 0 else f"Instance {i}"
            status = "✓" if self.enabled[i] else "✗"
            axes[i].set_title(f"{status} Layer {i}\n({layer_type})", fontsize=10)
            axes[i].axis('off')
            
            # Add border for disabled layers
            if not self.enabled[i]:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        
        # Show combined result
        combined = self.combine()
        axes[-1].imshow(combined)
        axes[-1].set_title("Combined Result", fontsize=12, fontweight='bold')
        axes[-1].axis('off')
        
        # Add border to combined result
        for spine in axes[-1].spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization: {save_path}")
        
        plt.show()
        
        return fig


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Create the tool
    tool = LayerCombiner(
        image_id="000000000285",
        layer_dir="/content/mulan_output"
    )
    
    # Show status
    tool.show_status()
    
    # Visualize everything
    tool.visualize(save_path="visualization_all_layers.png")
    
    # Combine and save
    combined = tool.combine()
    combined.save("combined_all_layers.png")
    print("✓ Saved: combined_all_layers.png")
    
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Disable Some Layers")
    print("=" * 60)
    
    # Disable layer 2
    tool.disable_layer(2)
    
    # Show updated status
    tool.show_status()
    
    # Visualize with layer 2 disabled
    tool.visualize(save_path="visualization_without_layer2.png")
    
    # Combine and save
    combined = tool.combine()
    combined.save("combined_without_layer2.png")
    print("✓ Saved: combined_without_layer2.png")
    
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multiple Variations")
    print("=" * 60)
    
    # Reset - enable all
    tool.enable_all()
    
    # Try different combinations
    variations = [
        ("all_layers", []),  # Empty list = enable all
        ("no_layer_1", [1]),
        ("no_layer_2", [2]),
        ("only_background", list(range(1, tool.get_num_layers()))),  # Disable all instances
    ]
    
    for name, layers_to_disable in variations:
        # Reset
        tool.enable_all()
        
        # Disable specified layers
        for layer_idx in layers_to_disable:
            tool.disable_layer(layer_idx)
        
        # Combine and save
        result = tool.combine()
        result.save(f"combined_{name}.png")
        print(f"✓ Saved: combined_{name}.png")
    
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Interactive Selection")
    print("=" * 60)
    
    # Start fresh
    tool2 = LayerCombiner(
        image_id="000000000285",
        layer_dir="/content/mulan_output"
    )
    
    # Disable all first
    tool2.disable_all()
    
    # Enable only what we want
    tool2.enable_layer(0)  # Background
    tool2.enable_layer(1)  # Instance 1
    tool2.enable_layer(3)  # Instance 3
    # Layer 2 stays disabled
    
    # Show what's enabled
    tool2.show_status()
    
    # Visualize
    tool2.visualize(save_path="visualization_selective.png")
    
    # Combine
    result = tool2.combine()
    result.save("combined_selective.png")
    print("✓ Saved: combined_selective.png")
    
    
    print("\n" + "=" * 60)
    print("Done! Check the saved images.")
    print("=" * 60)
    
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Natural Color Alteration (HSV-based)")
    print("=" * 60)
    
    # Create new tool
    tool3 = LayerCombiner(
        image_id="000000000285",
        layer_dir="/content/mulan_output"
    )
    
    # Original
    original = tool3.combine()
    original.save("color_original.png")
    print("✓ Saved: color_original.png")
    
    # Make warmer (more golden/reddish tones)
    tool3.set_layer_color(1, 'warmer', intensity=0.4)
    warmer = tool3.combine()
    warmer.save("color_warmer.png")
    print("✓ Saved: color_warmer.png - Natural warm shift")
    
    # Make cooler (more blue tones)
    tool3.reset_all_colors()
    tool3.set_layer_color(1, 'cooler', intensity=0.4)
    cooler = tool3.combine()
    cooler.save("color_cooler.png")
    print("✓ Saved: color_cooler.png - Natural cool shift")
    
    # Shift to specific hue (green) - natural looking
    tool3.reset_all_colors()
    tool3.set_layer_color(1, 'green', intensity=0.5)
    green = tool3.combine()
    green.save("color_green_natural.png")
    print("✓ Saved: color_green_natural.png - Natural green shift")
    
    # More saturated (vivid colors)
    tool3.reset_all_colors()
    tool3.set_layer_color(1, 'more_saturated', intensity=0.6)
    vivid = tool3.combine()
    vivid.save("color_vivid.png")
    print("✓ Saved: color_vivid.png - More vibrant")
    
    # Less saturated (washed out, grayish)
    tool3.reset_all_colors()
    tool3.set_layer_color(1, 'less_saturated', intensity=0.5)
    washed = tool3.combine()
    washed.save("color_washed.png")
    print("✓ Saved: color_washed.png - Less saturated")
    
    
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Advanced - Combine Color, Saturation, Brightness")
    print("=" * 60)
    
    tool4 = LayerCombiner(
        image_id="000000000285",
        layer_dir="/content/mulan_output"
    )
    
    # Shift to blue + boost saturation + brighten
    tool4.set_layer_color(1, 'blue', intensity=0.5, 
                          saturation_boost=0.3, brightness_adjust=0.2)
    vibrant_blue = tool4.combine()
    vibrant_blue.save("color_vibrant_blue.png")
    print("✓ Saved: color_vibrant_blue.png")
    
    # Golden hour effect: warmer + more saturated + slightly brighter
    tool4.reset_all_colors()
    tool4.set_layer_color(0, 'warmer', intensity=0.3,
                          saturation_boost=0.2, brightness_adjust=0.1)
    tool4.set_layer_color(1, 'orange', intensity=0.3,
                          saturation_boost=0.2)
    golden_hour = tool4.combine()
    golden_hour.save("color_golden_hour.png")
    print("✓ Saved: color_golden_hour.png - Golden hour effect")
    
    # Underwater effect: cyan/blue + less saturation + darker
    tool4.reset_all_colors()
    tool4.set_layer_color(0, 'cyan', intensity=0.5,
                          saturation_boost=-0.2, brightness_adjust=-0.1)
    tool4.set_layer_color(1, 'blue', intensity=0.4,
                          saturation_boost=-0.3)
    underwater = tool4.combine()
    underwater.save("color_underwater.png")
    print("✓ Saved: color_underwater.png - Underwater effect")
    
    
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Natural Color Harmony")
    print("=" * 60)
    
    tool5 = LayerCombiner(
        image_id="000000000285",
        layer_dir="/content/mulan_output"
    )
    
    # Complementary colors that work together
    # Background: warm sunset
    # Instances: cooler tones for contrast
    tool5.set_layer_color(0, 'orange', intensity=0.3, saturation_boost=0.1)
    tool5.set_layer_color(1, 'cyan', intensity=0.4)
    tool5.set_layer_color(2, 'blue', intensity=0.3)
    harmony1 = tool5.combine()
    harmony1.save("color_harmony_sunset.png")
    print("✓ Saved: color_harmony_sunset.png - Warm/cool harmony")
    
    # Monochromatic (different shades of same hue)
    tool5.reset_all_colors()
    tool5.set_layer_color(0, 'blue', intensity=0.3, brightness_adjust=-0.1)
    tool5.set_layer_color(1, 'blue', intensity=0.5)
    tool5.set_layer_color(2, 'cyan', intensity=0.4, brightness_adjust=0.1)
    harmony2 = tool5.combine()
    harmony2.save("color_harmony_mono.png")
    print("✓ Saved: color_harmony_mono.png - Monochromatic blue")
    
    
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Direct Hue Control (0-360)")
    print("=" * 60)
    
    tool6 = LayerCombiner(
        image_id="000000000285",
        layer_dir="/content/mulan_output"
    )
    
    # Shift to specific hue values
    # 0=red, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta
    tool6.set_layer_color(1, 120, intensity=0.6)  # Shift to green (120°)
    hue_green = tool6.combine()
    hue_green.save("color_hue_120_green.png")
    print("✓ Saved: color_hue_120_green.png - Hue 120° (green)")
    
    tool6.reset_all_colors()
    tool6.set_layer_color(1, 300, intensity=0.6)  # Shift to magenta (300°)
    hue_magenta = tool6.combine()
    hue_magenta.save("color_hue_300_magenta.png")
    print("✓ Saved: color_hue_300_magenta.png - Hue 300° (magenta)")
    
    
    print("\n" + "=" * 60)
    print("All examples done!")
    print("Color changes are NATURAL - texture and gradients preserved!")
    print("Uses HSV color space for realistic transformations.")
    print("=" * 60)
