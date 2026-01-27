import sys
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract as pt
from PIL import Image
import os

# --- CONFIGURATION ---
if len(sys.argv) > 1:
    PDF_PATH = sys.argv[1]
else:
    print("Usage: python parser.py <path_to_pdf>")
    sys.exit(1)

OUTPUT_PDF_NAME = "debug_visuals.pdf"
OUTPUT_SKEW_PDF_NAME = "debug_skew.pdf"
OUTPUT_TEXT_NAME = "extracted_output.txt"

DPI = 300
MIN_WIDTH_INCH = 0.1
MIN_HEIGHT_INCH = 0.1
min_w_pixels = int(MIN_WIDTH_INCH * DPI)
min_h_pixels = int(MIN_HEIGHT_INCH * DPI)
scale_factor = (DPI / 300) ** 2


# --- HELPER FUNCTIONS ---

def is_garbage_text(text):
    raw = text.strip()
    if not raw: return True
    alnum = sum(c.isalnum() for c in raw)
    total = len(raw)
    if total < 5:
        return (alnum / total) < 0.60
    return (alnum / total) < 0.40


def fix_orientation(image):
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
    """
    Returns: (angle, debug_image_bgr)
    """
    img_copy = image.copy()
    if len(img_copy.shape) == 3:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_copy, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Prepare Debug Image
    debug_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if lines is None:
        return 0.0, debug_vis

    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if -45 < angle < 45:
            angles.append(angle)
            cv2.line(debug_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green
        else:
            cv2.line(debug_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red

    num_lines = len(angles)
    if num_lines < 10:
        return 0.0, debug_vis

    std_dev = np.std(angles)
    if std_dev > 2.5:
        return 0.0, debug_vis

    return np.median(angles), debug_vis


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


def clean_heavy_grain(binary_image, min_area=int(150 * scale_factor)):
    inverted = cv2.bitwise_not(binary_image)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    clean_inverted = np.zeros_like(inverted)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            clean_inverted[labels == i] = 255
    return cv2.bitwise_not(clean_inverted)


def clean_digital_noise(image, d=5, sigmaColor=25, sigmaSpace=25):
    """
    Applies Bilateral Filtering with customizable strength.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    denoised = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


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

        # Geometric Filters
        aspect = w / float(h)
        if aspect > 50: continue
        if h / float(w) > 50: continue

        roi = clean_inverted[y:y + h, x:x + w]
        pixel_density = cv2.countNonZero(roi) / (w * h)
        if pixel_density > 0.90: continue
        if pixel_density < 0.01: continue

        blocks.append((x, y, w, h))

    row_band = kh
    return sorted(blocks, key=lambda b: ((b[1] // row_band) * row_band, b[0]))


def remove_nested_blocks(blocks):
    if not blocks: return []
    rem = set()
    for i in range(len(blocks)):
        for j in range(len(blocks)):
            if i == j: continue

            xi, yi, wi, hi = blocks[i]
            xj, yj, wj, hj = blocks[j]

            area_i = wi * hi
            area_j = wj * hj

            if area_i > area_j: continue

            inter_x_min = max(xi, xj)
            inter_y_min = max(yi, yj)
            inter_x_max = min(xi + wi, xj + wj)
            inter_y_max = min(yi + hi, yj + hj)

            inter_w = max(0, inter_x_max - inter_x_min)
            inter_h = max(0, inter_y_max - inter_y_min)

            intersection_area = inter_w * inter_h
            if intersection_area == 0: continue

            overlap_ratio = intersection_area / area_i

            if overlap_ratio > 0.50:
                rem.add(i)

    return [blocks[i] for i in range(len(blocks)) if i not in rem]


def filter_giant_blocks(blocks, img_w, img_h):
    if len(blocks) <= 1: return blocks
    return [b for b in blocks if (b[2] * b[3]) / (img_w * img_h) <= 0.85]


def is_dense_graphic(roi_binary):
    if roi_binary.size == 0: return False
    return (cv2.countNonZero(cv2.bitwise_not(roi_binary)) / roi_binary.size) > 0.45


def get_ocr_confidence(ocr_data_dict):
    """Calculates average confidence from pytesseract data dict."""
    confs = [int(c) for c in ocr_data_dict['conf'] if c != '-1']
    if not confs: return 0
    return sum(confs) / len(confs)


# --- MAIN EXECUTION ---

skew_summary = []
debug_images_list = []
debug_skew_list = []

doc = fitz.open(PDF_PATH)
full_document_text = ""

print(f"Processing PDF: {PDF_PATH} ({len(doc)} pages)")

for page_num, page in enumerate(doc):
    print(f"\n--- Processing Page {page_num + 1} ---")

    pix = page.get_pixmap(dpi=DPI, alpha=False)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))

    if pix.n == 1:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        img_gray = img_array
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # --- GATEKEEPER ---
    embedded_text_check = page.get_text("text")
    has_digital_layer = False
    if len(embedded_text_check) > 50 and not is_garbage_text(embedded_text_check):
        has_digital_layer = True

    fast_lane_success = False

    # --- FAST LANE ---
    if has_digital_layer:
        print("   >>> Digital Layer Detected. Attempting Fast Lane...")

        # 1. SKEW CORRECTION
        fast_skew, fast_debug_img = get_skew_angle(img_gray)

        if abs(fast_skew) > 0.5:
            print(f"   >>> [Fast Lane] Detected Skew {fast_skew:.2f}°. Correcting...")
            img_bgr = rotate_image(img_bgr, fast_skew)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            if fast_debug_img is not None:
                cv2.putText(fast_debug_img, f"Page {page_num + 1} (Fast): Skew {fast_skew:.2f}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                debug_skew_list.append(Image.fromarray(cv2.cvtColor(fast_debug_img, cv2.COLOR_BGR2RGB)))
            skew_summary.append((page_num + 1, fast_skew, "Fast Lane Correction"))
        else:
            skew_summary.append((page_num + 1, 0.0, "Fast Lane"))

        # 2. NOISE CORRECTION (Inside Fast Lane)
        if is_page_dirty(img_gray):
            print("   >>> [Fast Lane] Noise detected. Applying Gentle Bilateral (5, 25, 25)...")
            img_bgr = clean_digital_noise(img_bgr, d=5, sigmaColor=25, sigmaSpace=25)

        # # DEBUG PRE-OCR
        # debug_ocr_input = img_bgr.copy()
        # cv2.putText(debug_ocr_input, "DEBUG: FAST LANE INPUT (Pre-OCR)", (50, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        # debug_images_list.append(Image.fromarray(cv2.cvtColor(debug_ocr_input, cv2.COLOR_BGR2RGB)))

        # --- PROCEED TO OCR ---
        try:
            fast_text = pt.image_to_string(img_bgr, config='--psm 3')
        except:
            fast_text = ""

        if len(fast_text.strip()) > 50 and not is_garbage_text(fast_text):
            print("   >>> SUCCESS: Fast Lane OCR extracted text.")
            full_document_text += f"--- PAGE {page_num + 1} (FAST LANE) ---\n\n{fast_text}\n" + "=" * 50 + "\n"

            cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 80), (0, 200, 0), -1)
            cv2.putText(img_bgr, "FAST LANE: VISUAL OCR", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            debug_images_list.append(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))

            fast_lane_success = True
            continue
        else:
            print("   >>> Fast Lane produced garbage. Fallback to Robust Lane.")

    else:
        print("   >>> No Digital Layer found. FORCING Robust Pipeline.")

    # --- ROBUST PIPELINE ---

    # 1. Orientation Fix
    img_gray = fix_orientation(img_gray)

    # 2. Skew Detection
    initial_skew, skew_debug_img = get_skew_angle(img_gray)

    if skew_debug_img is not None:
        cv2.putText(skew_debug_img, f"Page {page_num + 1}: Skew {initial_skew:.2f}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        debug_skew_list.append(Image.fromarray(cv2.cvtColor(skew_debug_img, cv2.COLOR_BGR2RGB)))

    is_tilted = abs(initial_skew) > 0.5
    skew_summary.append((page_num + 1, initial_skew, "Corrected" if is_tilted else "Ignored"))
    print(f"      [Robust Start] Skew Detected: {initial_skew:.2f}°")

    # 3. Rotate if needed
    if is_tilted:
        img_gray = rotate_image(img_gray, initial_skew)
    else:
        skew_check, _ = get_skew_angle(img_gray)
        if abs(skew_check) > 0.1 and abs(skew_check) < 2.0:
            img_gray = rotate_image(img_gray, skew_check)

    # 4. PROCESSING LOGIC (CLEAN vs DIRTY)
    is_dirty = is_page_dirty(img_gray)

    # Initialize variables to avoid scope errors
    is_clean_mode = False
    clean_text_psm3 = ""
    clean_conf_psm3 = 0.0
    data_psm3 = None  # Initialized to None for safety

    if is_dirty:
        print("      [Robust] Mode: DIRTY -> Using Dual Pipeline (Aggressive Layout / Gentle OCR)")

        img_gray=cv2.GaussianBlur(img_gray, (3, 3), 0)  # anti-dithering fix
        # A. LAYOUT PATH (Aggressive: 7, 55, 55)
        blur_layout = cv2.bilateralFilter(img_gray, 7, 55, 55)
        thresh_layout_base = cv2.adaptiveThreshold(blur_layout, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                   15, 15)
        thresh_layout = clean_heavy_grain(thresh_layout_base, min_area=50)

        # B. OCR PATH (Gentle: 5, 25, 25)
        blur_ocr = cv2.bilateralFilter(img_gray, 5, 25, 25)
        thresh_ocr = cv2.adaptiveThreshold(blur_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 15)
        healed_ocr_img = cv2.erode(thresh_ocr, np.ones((2, 2), np.uint8), iterations=1)

    else:
        print("      [Robust] Mode: CLEAN -> Using Standard Gaussian Pipeline")
        is_clean_mode = True

        # --- LOGIC FROM SECOND CODE ---
        # Uses Standard Gaussian Blur (3,3) -> OTSU
        gray_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        thresh_gentle = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        thresh_layout = thresh_gentle
        healed_ocr_img = thresh_gentle

    # 5. Sync Crop
    h, w = healed_ocr_img.shape
    border = 20
    healed_ocr_img = cv2.copyMakeBorder(healed_ocr_img[border:h - border, border:w - border], border, border, border,
                                        border, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    thresh_layout = cv2.copyMakeBorder(thresh_layout[border:h - border, border:w - border], border, border, border,
                                       border, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    crop_coords = get_crop_coords(healed_ocr_img)
    if crop_coords:
        x1, y1, x2, y2 = crop_coords
        healed_ocr_img = healed_ocr_img[y1:y2, x1:x2]
        thresh_layout = thresh_layout[y1:y2, x1:x2]

    # --- NEW: OPTIONAL PSM3 PATH FOR CLEAN MODE ---
    if is_clean_mode:
        print("      [Robust Clean] Running Parallel OCR (PSM 3)...")
        try:
            # Run PSM 3 on the whole cropped image
            data_psm3 = pt.image_to_data(healed_ocr_img, config="--psm 3", output_type=pt.Output.DICT)
            clean_text_psm3 = pt.image_to_string(healed_ocr_img, config="--psm 3")
            clean_conf_psm3 = get_ocr_confidence(data_psm3)
            print(f"      [Robust Clean] PSM 3 Confidence: {clean_conf_psm3:.2f}")
        except:
            clean_conf_psm3 = 0.0

    # 6. Layout Analysis (Default Path A)
    edges = cv2.Canny(thresh_layout, 50, 150)
    layout_map = cv2.bitwise_not(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
    blocks = get_sorted_text_blocks(layout_map, min_w_pixels, min_h_pixels, DPI)
    blocks = filter_giant_blocks(blocks, healed_ocr_img.shape[1], healed_ocr_img.shape[0])
    blocks = remove_nested_blocks(blocks)

    print(f"      [Robust] Found {len(blocks)} blocks.")

    # 7. Extraction & Confidence Comparison
    vis_image_bgr = cv2.cvtColor(healed_ocr_img, cv2.COLOR_GRAY2BGR)

    # Collect text and confidence for Path A (Layout Analysis)
    layout_text_full = ""
    layout_conf_sum = 0
    layout_conf_count = 0
    block_results = []

    manual_block_count = len(blocks)
    forced_psm3 = False

    # --- UPDATED: INSTANT SWITCH LOGIC ---
    # If blocks > 50 in Clean Mode, skip Manual OCR completely
    if is_clean_mode and manual_block_count > 50:
        print(
            f"      [Robust Clean] Manual Layout Fragmented ({manual_block_count} blocks > 50). Instantly switching to PSM 3.")
        forced_psm3 = True
    else:
        # Run Manual OCR Loop (only if not forced to skip)
        for i, (x, y, w, h) in enumerate(blocks):
            roi = healed_ocr_img[y:y + h, x:x + w]
            if is_dense_graphic(roi):
                block_results.append((i, x, y, w, h, "image", "", 0))
                continue

            try:
                data_box = pt.image_to_data(roi, config="--oem 3 --psm 6", output_type=pt.Output.DICT)
                text_box = pt.image_to_string(roi, config="--oem 3 --psm 6")
                conf_box = get_ocr_confidence(data_box)
            except:
                text_box = ""
                conf_box = 0

            if is_garbage_text(text_box):
                block_results.append((i, x, y, w, h, "diagram", "", 0))
            else:
                block_results.append((i, x, y, w, h, "text", text_box, conf_box))
                layout_text_full += text_box + "\n\n"
                layout_conf_sum += conf_box
                layout_conf_count += 1

    # Calculate Average Confidence for Layout Path
    layout_avg_conf = (layout_conf_sum / layout_conf_count) if layout_conf_count > 0 else 0.0

    if is_clean_mode:
        print(
            f"      [Robust Clean] Comparison -> PSM 3 Conf: {clean_conf_psm3:.2f} vs Layout Conf: {layout_avg_conf:.2f}")

    # DECISION: SWITCH TO PSM 3?
    use_psm3 = False
    reason = ""

    # Only run logic if Clean Mode (prevents Dirty Mode access to PSM3 data)
    if is_clean_mode:
        try:
            if data_psm3:
                psm3_block_count = len(set(data_psm3['block_num']))
            else:
                psm3_block_count = 1
        except:
            psm3_block_count = 1

        print(
            f"      [Robust Clean] Structure -> Manual Blocks: {manual_block_count} | PSM 3 Blocks: {psm3_block_count}")

        # Condition 0: Forced Fragmentation (Immediate Switch)
        if forced_psm3:
            use_psm3 = True
            reason = f"Manual Layout Fragmented (>50 blocks)"

        # Condition 1: Confidence dominance (Existing)
        elif clean_conf_psm3 > (layout_avg_conf + 8):
            use_psm3 = True
            reason = f"Higher Confidence (+8)"

        # Condition 2: Structural Superiority (New)
        elif manual_block_count <= 3 and psm3_block_count > min(manual_block_count, 3):
            if clean_conf_psm3 > 50:
                use_psm3 = True
                reason = "Better Layout Detection (Multi-block vs Single)"

    # --- APPLY DECISION ---
    if use_psm3:
        print(f"      [Robust Clean] SWITCHING to PSM 3 Output. Reason: {reason}")
        final_page_text = f"--- PAGE {page_num + 1} (ROBUST CLEAN - PSM 3) ---\n\n{clean_text_psm3}\n" + "=" * 50 + "\n"

        cv2.rectangle(vis_image_bgr, (0, 0), (vis_image_bgr.shape[1], 50), (0, 255, 0), -1)
        cv2.putText(vis_image_bgr, f"SELECTED: PSM 3 ({reason})", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    else:
        # Stick to Manual Layout (Default)
        if is_clean_mode:
            print("      [Robust Clean] Keeping Manual Layout Output.")

        final_page_text = f"--- PAGE {page_num + 1} (ROBUST PIPELINE) ---\n\n"
        for idx, x, y, w, h, b_type, text, conf in block_results:
            if b_type == "image":
                final_page_text += f"[IMAGE {idx}]\n\n"
                cv2.rectangle(vis_image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 3)
            elif b_type == "diagram":
                final_page_text += f"[DIAGRAM {idx}]\n\n"
                cv2.rectangle(vis_image_bgr, (x, y), (x + w, y + h), (0, 165, 255), 3)
            else:
                final_page_text += text + "\n\n"
                cv2.rectangle(vis_image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(vis_image_bgr, str(idx), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        final_page_text += "=" * 50 + "\n"

    full_document_text += final_page_text
    debug_images_list.append(Image.fromarray(cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)))

# --- SKEW SUMMARY PRINT ---
print("\n" + "=" * 50)
print("             SKEW CORRECTION SUMMARY")
print("=" * 50)
print(f"{'Page':<10} | {'Skew Angle':<15} | {'Action Taken'}")
print("-" * 50)
for p_num, angle, action in skew_summary:
    print(f"{p_num:<10} | {angle:<15.2f} | {action}")
print("=" * 50)

# --- SAVE OUTPUTS ---
with open(OUTPUT_TEXT_NAME, "w", encoding="utf-8") as f:
    f.write(full_document_text)
print(f"\n>>> Saved text to: {OUTPUT_TEXT_NAME}")

# Save Layout Analysis PDF
if debug_images_list:
    debug_images_list[0].save(
        OUTPUT_PDF_NAME, "PDF", resolution=100.0, save_all=True, append_images=debug_images_list[1:]
    )
    print(f">>> Saved visual debug PDF to: {OUTPUT_PDF_NAME}")

# Save SKEW PDF
if debug_skew_list:
    debug_skew_list[0].save(
        OUTPUT_SKEW_PDF_NAME, "PDF", resolution=100.0, save_all=True, append_images=debug_skew_list[1:]
    )
    print(f">>> Saved SKEW DEBUG PDF to: {OUTPUT_SKEW_PDF_NAME}")
