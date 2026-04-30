import os
import json
from PIL import Image, ImageDraw, ImageFont
from config import IMAGE_DIR, DATASET_PATH, FONT_DIR, FONTS, BREEDS_INFO, IMG_SIZE
import textwrap  
from collections import Counter
import random

# === Fallback: pixel-level word wrap for edge cases ===
def wrap_text_fallback(text, font, max_width, draw):
    """
    When the middle-split strategy fails (text still too wide), wrap into multiple lines.
    Returns a list of line strings.
    """
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line_words = current_line + [word]
        test_line_str = " ".join(test_line_words)
        # Older Pillow: no getlength
        try:
            line_width = font.getlength(test_line_str)
        except AttributeError:
            bbox = draw.textbbox((0, 0), test_line_str, font=font)
            line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line_words
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def draw_text_smart(draw, text, font, img_size, pos_ratio, margin=10, fill=(0, 0, 0)):
    """
    Line-by-line layout and drawing to avoid overlapping glyphs.
    """
    W, H = img_size
    px, py = pos_ratio
    max_width = W - (2 * margin)

    # Line spacing: 20% of font size
    line_spacing = int(font.size * 0.2) if hasattr(font, 'size') else 5

    # === Step 1: Choose line breaks ===
    lines = []
    
    # Prefer a single line if it fits
    try:
        text_w = font.getlength(text)
    except AttributeError:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]

    if text_w <= max_width:
        lines = [text]
    else:
        # Too wide: try splitting down the middle (two halves)
        words = text.split(" ")
        if len(words) > 1:
            mid = len(words) // 2
            line1 = " ".join(words[:mid])
            line2 = " ".join(words[mid:])
            
            # Width check after split
            try:
                w1 = font.getlength(line1)
                w2 = font.getlength(line2)
            except AttributeError:
                b1 = draw.textbbox((0, 0), line1, font=font)
                b2 = draw.textbbox((0, 0), line2, font=font)
                w1, w2 = b1[2]-b1[0], b2[2]-b2[0]
            
            if w1 <= max_width and w2 <= max_width:
                lines = [line1, line2]
            else:
                # Split failed: use fallback wrap
                lines = wrap_text_fallback(text, font, max_width, draw)
        else:
            # Single long token: one line; clamping keeps it on canvas
            lines = [text]

    # === Step 2: Line metrics and total height ===
    line_dims = []  # (width, height) per line
    total_height = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        line_dims.append((lw, lh))
        total_height += lh
    
    # Include line spacing in total height
    total_height += (len(lines) - 1) * line_spacing if len(lines) > 0 else 0

    # === Step 3: Start Y so the block is vertically centered on the anchor ===
    target_y = H * py
    start_y = target_y - (total_height / 2)

    # === Step 4: Draw each line ===
    current_y = start_y
    drawn_bboxes = []  # bbox per drawn line

    for i, line in enumerate(lines):
        lw, lh = line_dims[i]
        
        # Horizontally center this line on the anchor
        target_x = W * px
        current_x = target_x - (lw / 2)

        # --- Clamp to margins ---
        current_x = max(margin, min(current_x, W - lw - margin))
        current_y_clamped = max(margin, min(current_y, H - lh - margin))
        
        draw.text((current_x, current_y_clamped), line, font=font, fill=fill)
        
        drawn_bboxes.append((current_x, current_y_clamped, current_x + lw, current_y_clamped + lh))

        # Next line Y
        current_y += lh + line_spacing

    # Overall bounding box (left, top, right, bottom)
    if drawn_bboxes:
        min_x = min([b[0] for b in drawn_bboxes])
        min_y = min([b[1] for b in drawn_bboxes])
        max_x = max([b[2] for b in drawn_bboxes])
        max_y = max([b[3] for b in drawn_bboxes])
        return (min_x, min_y, max_x, max_y)
    else:
        return (0,0,0,0)


def build_balanced_color_pool(n, colors, seed=None):
    """
    Build a list of length n with colors as evenly distributed as possible,
    then shuffle so order is random but counts stay balanced.
    """
    if seed is not None:
        random.seed(seed)

    k = len(colors)
    base = n // k
    rem = n % k

    pool = []
    # Base count per color
    for name, rgb in colors:
        pool.extend([(name, rgb)] * base)
    # Assign remainder to the first `rem` colors; shuffle still randomizes order
    for idx in range(rem):
        pool.append(colors[idx])

    random.shuffle(pool)
    return pool


def generate_images():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    dataset = []
    
    combinations = [
        (30, 0.5, 0.5), (35, 0.5, 0.5), (40, 0.5, 0.5),  # center, varying sizes
        (35, 0.5, 0.8), (35, 0.5, 0.2),  # near top / bottom
    ]


    cursive_COLORS = [
        ("red",    (255, 0, 0)),
        ("blue",   (0, 0, 255)),
        ("green",  (0, 160, 0)),
        ("purple", (128, 0, 128)),
        ("black",  (0, 0, 0)),
        
    ]

    print(">>> Starting Generation...")
    
    # Total cursive images (for strictly balanced color assignment)
    num_breeds = len(BREEDS_INFO)
    num_cursive_fonts = len(FONTS.get("cursive", []))
    num_positions = len(combinations)

    total_cursive_images = num_breeds * num_cursive_fonts * num_positions

    # Balanced pool in random order; pop() one entry per cursive image
    cursive_color_pool = build_balanced_color_pool(
        total_cursive_images,
        cursive_COLORS,
        seed=42  # Remove for different runs each time; keep for reproducibility
    )

    # Optional: per-image color map for a sidecar file
    color_records = []

    img_id = 0
    for breed_name, animal_type in BREEDS_INFO:
        for font_category, font_list in FONTS.items():
            for font_name in font_list:
                font_path = os.path.join(FONT_DIR, font_category, font_name)
                if not os.path.exists(font_path):
                    print(f"ERROR: Font not found at {font_path}")
                    continue

                for i, (size, px, py) in enumerate(combinations):
                    img = Image.new('RGB', IMG_SIZE, (255, 255, 255))
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype(font_path, size)
                    except:
                        font = ImageFont.load_default()

                    # Colors: balanced pool for "cursive"; otherwise black
                    if font_category == "cursive":
                        color_name, color_rgb = cursive_color_pool.pop()
                    else:
                        color_name, color_rgb = "black", (0, 0, 0)

                    final_bbox = draw_text_smart(
                        draw,
                        breed_name,
                        font,
                        IMG_SIZE,
                        (px, py),
                        margin=10,
                        fill=color_rgb,
                    )

                    safe_breed = breed_name.replace(" ", "_")
                    filename = f"{safe_breed}_{font_category}_{font_name[:-4]}_{i}.png"
                    filepath = os.path.join(IMAGE_DIR, filename)
                    img.save(filepath)

                    dataset.append({
                        "id": img_id,
                        "image_path": filepath,
                        "breed": breed_name,
                        "animal_type": animal_type,
                        "font_category": font_category,
                        "font_name": font_name,
                        "size": size,
                        "position_id": i,
                        "text_color_name": color_name,
                        "text_color_rgb": list(color_rgb),
                    })

                    color_records.append({
                        "id": img_id,
                        "image_path": filepath,
                        "font_category": font_category,
                        "text_color_name": color_name,
                        "text_color_rgb": list(color_rgb),
                    })

                    img_id += 1


    with open(DATASET_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f">>> Generation Done. {len(dataset)} images created.")

    # Summary: color counts (non-black only for cursive)
    color_count = Counter([d["text_color_name"] for d in dataset if d["font_category"] == "cursive"])
    color_summary_path = os.path.splitext(DATASET_PATH)[0] + "_cursive_color_summary.json"
    with open(color_summary_path, "w") as f:
        json.dump(color_count, f, indent=2)

    # Optional: full per-image color map
    color_map_path = os.path.splitext(DATASET_PATH)[0] + "_color_map.json"
    with open(color_map_path, "w") as f:
        json.dump(color_records, f, indent=2)

    print(f">>> Generation Done. {len(dataset)} images created.")
    print(f">>> cursive color summary saved to: {color_summary_path}")
    print(f">>> Color map saved to: {color_map_path}")

if __name__ == "__main__":
    generate_images()