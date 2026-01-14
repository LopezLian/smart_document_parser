import sys
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract as pt
from PIL import Image
import os

if len(sys.argv) > 1:
    PDF_PATH = sys.argv[1]
else:
    # If they didn't provide a file, just exit or use a default for testing
    print("Usage: python parser.py <path_to_pdf>")
    sys.exit(1)
OUTPUT_PDF_NAME = "debug_visuals.pdf"
OUTPUT_TEXT_NAME = "extracted_output.txt"

DPI = 300
MIN_WIDTH_INCH = 0.1
MIN_HEIGHT_INCH = 0.1
min_w_pixels = int(MIN_WIDTH_INCH * DPI)
min_h_pixels = int(MIN_HEIGHT_INCH * DPI)


# --- HELPER FUNCTIONS ---

def is_garbage_text(text):
    """
    Checks if a BLOCK of text is garbage.
    Accepts short text (titles) but rejects pure noise.
    """
    raw = text.strip()
    if not raw: return True

    alnum = sum(c.isalnum() for c in raw)
    total = len(raw)

    # Tiny blocks (Page numbers, "Fig 1") need high purity
    if total < 5:
        return (alnum / total) < 0.60

    # Normal blocks just need to be readable
    return (alnum / total) < 0.40


def fix_orientation(image):
    """Detects and fixes 90/180/270 degree rotation."""
    MIN_CONFIDENCE = 5.0
    h, w = image.shape
    crop = image[h // 2 - h // 4: h // 2 + h // 4, w // 2 - w // 4: w // 2 + w // 4]
    _, binary_crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        custom_config = f'--dpi {DPI} --psm 0'
        osd = pt.image_to_osd(binary_crop, config=custom_config, output_type=pt.Output.DICT)
        if osd['rotate'] != 0 and osd['orientation_conf'] > MIN_CONFIDENCE:
            print(f"      [Robust] Correcting Orientation: {osd['rotate']}°")
            if osd['rotate'] == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif osd['rotate'] == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif osd['rotate'] == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except:
        pass
    return image

def get_skew_angle(image):
    img_copy = image.copy()
    if len(img_copy.shape) == 3:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_copy, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return 0.0

    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if -45 < angle < 45:
            angles.append(angle)

    num_lines = len(angles)

    if num_lines < 10:
        return 0.0

    std_dev = np.std(angles)

    if std_dev > 2.0:
        return 0.0

    print(f"Variance: {std_dev:.2f}")
    return np.median(angles)

def rotate_image(image, angle):
    if abs(angle) < 0.1: return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def is_page_dirty(gray_image):
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    noise_ratio = np.sum((gray_image > 50) & (gray_image < 240)) / gray_image.size

    return np.argmax(hist) < 253 or noise_ratio > 0.15


def clean_heavy_grain(binary_image, min_area=50):
    """Nuclear option for Layout Map."""
    inverted = cv2.bitwise_not(binary_image)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    clean_inverted = np.zeros_like(inverted)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            clean_inverted[labels == i] = 255
    return cv2.bitwise_not(clean_inverted)


def get_crop_coords(image):
    inverted = cv2.bitwise_not(image)
    coords = cv2.findNonZero(inverted)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    pad = 10
    h_img, w_img = image.shape
    return (max(0, x - pad), max(0, y - pad), min(w_img, x + w + pad), min(h_img, y + h + pad))


def get_sorted_text_blocks(layout_map, min_w, min_h, dpi):
    inverted = cv2.bitwise_not(layout_map)
    # Dynamic Safety Frame
    safety_margin = max(10, int(0.1 * dpi))
    inverted[0:safety_margin, :] = 0
    inverted[inverted.shape[0] - safety_margin:, :] = 0
    inverted[:, 0:safety_margin] = 0
    inverted[:, inverted.shape[1] - safety_margin:] = 0

    cnts_raw, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_inverted = np.zeros_like(inverted)


    for c in cnts_raw:
        if cv2.contourArea(c) > 50:
            cv2.drawContours(clean_inverted, [c], -1, 255, -1)

    kw, kh = int(0.08 * dpi), int(0.25 * dpi)
    if kw % 2 == 0: kw += 1
    if kh % 2 == 0: kh += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(clean_inverted, kernel, iterations=1)

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h: continue
        if cv2.countNonZero(clean_inverted[y:y + h, x:x + w]) / (w * h) < 0.01: continue
        blocks.append((x, y, w, h))

    row_band = kh
    return sorted(blocks, key=lambda b: ((b[1] // row_band) * row_band, b[0]))


def remove_nested_blocks(blocks):
    if not blocks: return []
    rem = set()
    for i in range(len(blocks)):
        for j in range(len(blocks)):
            if i == j: continue
            if (blocks[i][0] >= blocks[j][0] - 5 and blocks[i][1] >= blocks[j][1] - 5 and
                    blocks[i][0] + blocks[i][2] <= blocks[j][0] + blocks[j][2] + 5 and
                    blocks[i][1] + blocks[i][3] <= blocks[j][1] + blocks[j][3] + 5):
                if blocks[i][2] * blocks[i][3] < blocks[j][2] * blocks[j][3]: rem.add(i)
    return [blocks[i] for i in range(len(blocks)) if i not in rem]


def filter_giant_blocks(blocks, img_w, img_h):
    if len(blocks) <= 1: return blocks
    return [b for b in blocks if (b[2] * b[3]) / (img_w * img_h) <= 0.85]


def is_dense_graphic(roi_binary):
    if roi_binary.size == 0: return False
    return (cv2.countNonZero(cv2.bitwise_not(roi_binary)) / roi_binary.size) > 0.45


# --- MAIN EXECUTION ---

doc = fitz.open(PDF_PATH)
full_document_text = ""
debug_images_list = []

print(f"Processing PDF: {PDF_PATH} ({len(doc)} pages)")

for page_num, page in enumerate(doc):
    print(f"\n--- Processing Page {page_num + 1} ---")

    # 1. Get Image
    pix = page.get_pixmap(dpi=DPI, alpha=False)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))

    if pix.n == 1:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        img_gray = img_array
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # =================================================================
    # THE GATEKEEPER: PRE-FLIGHT CHECKS
    # =================================================================
    # Check 1: Alignment (Crucial for Photos)
    initial_skew = get_skew_angle(img_gray)
    is_tilted = abs(initial_skew) > 0.5  # Zero Tolerance: > 0.5 degrees

    # Check 2: Noise (Crucial for Scans)
    is_dirty = is_page_dirty(img_gray)

    fast_lane_success = False

    # DECISION: ONLY Perfect (Straight & Clean) Pages get Fast Lane
    if not is_tilted and not is_dirty:
        print("   >>> Page is Straight (0.0°) and Clean. Attempting Fast Lane...")
        try:
            fast_text = pt.image_to_string(img_bgr, config='--psm 3')
        except:
            fast_text = ""

        # Check if Tesseract actually succeeded (Must be >50 chars AND valid text)
        if len(fast_text.strip()) > 50 and not is_garbage_text(fast_text):
            print("   >>> SUCCESS: Digital text extracted. Skipping Robust Pipeline.")
            full_document_text += f"--- PAGE {page_num + 1} (FAST LANE) ---\n\n{fast_text}\n" + "=" * 50 + "\n"

            # Draw Blue Banner for Fast Lane
            cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 80), (255, 0, 0), -1)
            cv2.putText(img_bgr, "FAST LANE: DIGITAL", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            debug_images_list.append(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))

            fast_lane_success = True
        else:
            print("   >>> Fast Lane Failed (Complex Layout or Empty). Switching to Robust...")
    else:
        reason = "Tilted" if is_tilted else f"Dirty/Noisy"
        print(f"   >>> {reason} Page Detected (Skew: {initial_skew:.2f}°). FORCING Robust Pipeline.")

    # =================================================================
    # ROBUST PIPELINE (The Specialist)
    # =================================================================
    if not fast_lane_success:

        # 1. Fix Orientation & Skew
        # We reuse the 'initial_skew' we calculated in the Gatekeeper!
        gray = fix_orientation(img_gray)

        if is_tilted:
            print(f"      [Robust] Correcting Skew: {initial_skew:.2f}°")
            gray = rotate_image(gray, initial_skew)
        else:
            # Re-check skew after orientation fix just to be safe
            skew_check = get_skew_angle(gray)
            if abs(skew_check) > 0.1:
                gray = rotate_image(gray, skew_check)

        # 2. Re-evaluate Clean/Dirty on the FIX image
        # (Sometimes rotating fixes edge artifacts that looked like noise)
        final_dirty_check = is_page_dirty(gray)

        if final_dirty_check:
            print("      [Robust] Mode: DIRTY (Hybrid Dual-Stream)")
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Stream 1: Gentle (OCR)
            thresh_gentle = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15,
                                                  15)

            # Stream 2: Nuclear (Layout)
            thresh_layout = clean_heavy_grain(thresh_gentle, min_area=50)

            # Heal OCR stream
            healed_ocr_img = cv2.erode(thresh_gentle, np.ones((2, 2), np.uint8), iterations=1)

        else:
            print("      [Robust] Mode: CLEAN (Otsu Single-Stream)")
            # This handles Digital files that failed Fast Lane (e.g. complex tables)
            gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh_gentle = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            thresh_layout = thresh_gentle
            healed_ocr_img = thresh_gentle

        # 3. Sync Crop
        h, w = healed_ocr_img.shape
        border = 20
        healed_ocr_img = cv2.copyMakeBorder(healed_ocr_img[border:h - border, border:w - border], border, border,
                                            border, border, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        thresh_layout = cv2.copyMakeBorder(thresh_layout[border:h - border, border:w - border], border, border, border,
                                           border, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        crop_coords = get_crop_coords(healed_ocr_img)
        if crop_coords:
            x1, y1, x2, y2 = crop_coords
            healed_ocr_img = healed_ocr_img[y1:y2, x1:x2]
            thresh_layout = thresh_layout[y1:y2, x1:x2]

        # 4. Detection
        edges = cv2.Canny(thresh_layout, 50, 150)
        layout_map = cv2.bitwise_not(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))

        blocks = get_sorted_text_blocks(layout_map, min_w_pixels, min_h_pixels, DPI)
        blocks = filter_giant_blocks(blocks, healed_ocr_img.shape[1], healed_ocr_img.shape[0])
        blocks = remove_nested_blocks(blocks)

        print(f"      [Robust] Found {len(blocks)} blocks.")

        # 5. Extraction
        vis_image_bgr = cv2.cvtColor(healed_ocr_img, cv2.COLOR_GRAY2BGR)
        page_text_content = f"--- PAGE {page_num + 1} (ROBUST PIPELINE) ---\n\n"

        for i, (x, y, w, h) in enumerate(blocks):
            roi = healed_ocr_img[y:y + h, x:x + w]

            if is_dense_graphic(roi):
                page_text_content += f"[IMAGE {i}]\n\n"
                cv2.rectangle(vis_image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 3)
                continue

            text = pt.image_to_string(roi, config="--oem 3 --psm 6")

            if is_garbage_text(text):
                page_text_content += f"[DIAGRAM {i}]\n\n"
                cv2.rectangle(vis_image_bgr, (x, y), (x + w, y + h), (0, 165, 255), 3)
            else:
                page_text_content += text + "\n\n"
                cv2.rectangle(vis_image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(vis_image_bgr, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        full_document_text += page_text_content + "\n" + "=" * 50 + "\n"

        vis_image_rgb = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)
        debug_images_list.append(Image.fromarray(vis_image_rgb))

# --- SAVE OUTPUTS ---
with open(OUTPUT_TEXT_NAME, "w", encoding="utf-8") as f:
    f.write(full_document_text)
print(f"\n>>> Saved text to: {OUTPUT_TEXT_NAME}")

if debug_images_list:
    debug_images_list[0].save(
        OUTPUT_PDF_NAME, "PDF", resolution=100.0, save_all=True, append_images=debug_images_list[1:]
    )
    print(f">>> Saved visual debug PDF to: {OUTPUT_PDF_NAME}")