
import sys
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract as pt
from PIL import Image
import os
import re

# --- NEW: IMPORT FOR SMART SCORING GATEKEEPER ---
try:
    from transformers import AutoTokenizer

    print("Loading InLegalBERT Dictionary for Smart Page Scoring...")
    legal_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
except ImportError:
    print("WARNING: 'transformers' not installed. Run 'pip install transformers'. Fallback scoring will be used.")
    legal_tokenizer = None

# --- CONFIGURATION ---
if len(sys.argv) > 1:
    PDF_PATH = sys.argv[1]
else:
    print("Usage: python parser.py <path_to_pdf>")
    sys.exit(1)

# --- DYNAMIC OUTPUT DIRECTORY SETUP ---
# Creates ./output/pdf_name/ to perfectly match the LightOn parser structure
PDF_BASENAME = os.path.basename(PDF_PATH)
PDF_FILENAME, _ = os.path.splitext(PDF_BASENAME)
OUTPUT_DIR = os.path.join("./output", PDF_FILENAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Point debug PDFs to save inside the new folder
OUTPUT_PDF_NAME = os.path.join(OUTPUT_DIR, "debug_visuals.pdf")
OUTPUT_SKEW_PDF_NAME = os.path.join(OUTPUT_DIR, "debug_skew.pdf")

DPI = 300
MIN_WIDTH_INCH = 0.1
MIN_HEIGHT_INCH = 0.1
min_w_pixels = int(MIN_WIDTH_INCH * DPI)
min_h_pixels = int(MIN_HEIGHT_INCH * DPI)
scale_factor = (DPI / 300) ** 2


# --- HELPER FUNCTIONS ---

def basic_text_cleanup(text):
    """
    Safer cleanup that preserves list markers (1., A.) and technical text.
    Uses 'Do No Harm' hyphenation to protect proper nouns and legal/tech jargon.
    """
    if not text:
        return ""

    # 1. THE FIX: Do No Harm Hyphenation
    # "cross-\nexamination" -> "cross- examination"
    text = re.sub(r'-\s*\n\s*', '- ', text)

    # 2. Remove pipes | often read from table borders
    text = text.replace('|', ' ')

    # 3. Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    # 4. Safer Line Filtering
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()

        # LOGIC: Only delete the line if it is short AND has NO letters/numbers.
        if len(stripped) < 4 and not any(c.isalnum() for c in stripped):
            continue

        clean_lines.append(line)

    text = "\n".join(clean_lines)

    # 5. Limit newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def is_garbage_text(text):
    raw = text.strip()
    if not raw: return True

    if len(raw) < 10: return True

    alnum = sum(c.isalnum() for c in raw)
    total = len(raw)
    if (alnum / total) < 0.50: return True

    non_ascii = sum(1 for c in raw if ord(c) > 127)
    if (non_ascii / total) > 0.20: return True

    if total > 20:
        vowels = set("aeiouyAEIOUY")
        vowel_count = sum(1 for c in raw if c in vowels)
        if (vowel_count / total) < 0.10: return True

    return False


def get_page_quality_score(page_text, page_conf):
    """
    Evaluates final extracted text.
    Uses a hybrid token-to-word ratio to catch margin garbage while preserving valid words.
    """
    if not page_text or len(page_text.strip()) == 0:
        return 999.0, "GARBAGE (Empty)"

    # 1. Strip out our script's structural markers
    raw_text = re.sub(r'\[IMAGE.*?\]|\[DIAGRAM.*?\]|--- PAGE.*?---|=+', '', page_text).strip()

    # 2. Strip URLs, DOIs, Emails, AND arXiv links
    raw_text = re.sub(r'http[s]?://\S+|www\.\S+|doi\.org/\S+|\S+@\S+|arXiv:\S+', '', raw_text)

    # 3. THE FIX: Neutralize grammatical "wrapper" punctuation.
    scoring_text = re.sub(r'[\[\]\(\)\,\:\;\"\']', ' ', raw_text)

    # 4. Create the word-count string (swapping the remaining gluing punctuation)
    clean_for_words = re.sub(r'[/\.\-_@&]', ' ', scoring_text)
    clean_for_words = re.sub(r'\s+', ' ', clean_for_words).strip()

    words = clean_for_words.split()
    word_count = len(words)

    if word_count < 10:
        return 999.0, "GARBAGE (No Text Extracted)"
    elif word_count < 45:
        return 999.0, f"GARBAGE (Sparse Text: {word_count} words - Likely Cover Page or Bad Extraction)"

    # 5. Get Token Count
    if legal_tokenizer:
        tokens = legal_tokenizer.tokenize(scoring_text)
        ratio = len(tokens) / word_count
    else:
        ratio = 1.0

    # --- ADJUSTED TWO-FACTOR ROUTING LOGIC ---
    if ratio > 1.80:
        return ratio, "GARBAGE (High Error Rate)"
    elif 1.50 < ratio <= 1.80:
        # The Gray Area Check
        if page_conf < 65:
            return ratio, f"GARBAGE (Borderline Ratio + Low Conf: {page_conf:.1f}%)"
        else:
            return ratio, f"CLEAN (Borderline Ratio + High Conf: {page_conf:.1f}%)"
    else:
        return ratio, "CLEAN (Good Extraction)"


def fix_orientation(image):
    MIN_CONFIDENCE = 5.0
    h, w = image.shape

    # UPGRADE: Resize entire image to 800px for lightning-fast, full-context OSD
    scale = 800.0 / max(h, w)
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        resized = image

    _, binary_resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        osd = pt.image_to_osd(binary_resized, config='--psm 0', output_type=pt.Output.DICT)
        if osd['rotate'] != 0 and osd['orientation_conf'] > MIN_CONFIDENCE:
            print(f"      [Robust] Correcting Orientation: {osd['rotate']}°")
            if osd['rotate'] == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif osd['rotate'] == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif osd['rotate'] == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return image


def get_skew_angle(image):
    """
    Returns: (angle, debug_image_bgr)
    """
    img_copy = image.copy()
    if len(img_copy.shape) == 3:
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_copy

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    debug_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    angles = []

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if -45 < angle < 45:
                angles.append(angle)
                cv2.line(debug_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            else:
                cv2.line(debug_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    num_lines = len(angles)

    if num_lines < 10 or (num_lines >= 10 and np.std(angles) > 2.5):
        print("      [Skew] HoughLines insufficient/noisy. Triggering Projection Profile Fallback...")

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = binary.shape
        center = (w // 2, h // 2)

        def get_variance(angle):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            projection = np.sum(rotated, axis=1)
            return np.var(projection)

        best_coarse_angle = 0.0
        max_var = 0.0
        for angle in np.arange(-5.0, 6.0, 1.0):
            variance = get_variance(angle)
            if variance > max_var:
                max_var = variance
                best_coarse_angle = angle

        best_final_angle = best_coarse_angle
        for angle in np.arange(best_coarse_angle - 1.0, best_coarse_angle + 1.1, 0.1):
            variance = get_variance(angle)
            if variance > max_var:
                max_var = variance
                best_final_angle = angle

        cv2.putText(debug_vis, f"Fallback Skew: {best_final_angle:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)

        return best_final_angle, debug_vis

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


def clean_digital_noise(image, h=10, templateWindowSize=7, searchWindowSize=21):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    denoised = cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=templateWindowSize,
                                        searchWindowSize=searchWindowSize)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


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
    Image.fromarray(dilated).show()

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h: continue

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
    confs = [int(c) for c in ocr_data_dict['conf'] if c != '-1']
    if not confs: return 0
    return sum(confs) / len(confs)


# --- MAIN EXECUTION ---

skew_summary = []
debug_images_list = []
debug_skew_list = []
pages_for_dots_ocr = []

doc = fitz.open(PDF_PATH)

print(f"Processing PDF: {PDF_PATH} ({len(doc)} pages)")
print(f"Saving extracted markdown files to: {OUTPUT_DIR}/")

for page_num, page in enumerate(doc):
    print(f"\n--- Processing Page {page_num + 1} ---")

    # Setup specific filename for this page to match LightOn parser
    page_md_filename = f"{PDF_FILENAME}_page_{page_num + 1}.md"
    page_md_path = os.path.join(OUTPUT_DIR, page_md_filename)

    pix = page.get_pixmap(dpi=DPI, alpha=False)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))

    if pix.n == 1:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        img_gray = img_array
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # --- TIER 1: GATEKEEPER / NATIVE DIGITAL LIE DETECTOR ---
    raw_native_text = page.get_text("text")
    native_text = basic_text_cleanup(raw_native_text)

    native_score, native_status = get_page_quality_score(native_text, 50.0)
    attempt_fast_lane = False

    if len(native_text) > 50 and not is_garbage_text(native_text) and "CLEAN" in native_status:
        print(f"   >>> [Gatekeeper] Pristine digital layer verified (Score: {native_score:.2f}). Bypassing OCR.")

        page_text_to_add = f"--- PAGE {page_num + 1} (TIER 1: NATIVE DIGITAL) ---\n\n{native_text}\n" + "=" * 50 + "\n"

        # Write output immediately to disk for this page
        with open(page_md_path, "w", encoding="utf-8") as f:
            f.write(page_text_to_add)

        cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 80), (255, 200, 0), -1)
        cv2.putText(img_bgr, "TIER 1: NATIVE DIGITAL", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        debug_images_list.append(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))

        skew_summary.append((page_num + 1, 0.0, "Native Digital (Skipped)"))
        continue  # <--- Skip all OCR

    else:
        if len(raw_native_text.strip()) > 50:
            print(f"   >>> [Gatekeeper] Bogus digital layer detected (Score: {native_score:.2f} -> {native_status}).")
            attempt_fast_lane = True
        else:
            print("   >>> [Gatekeeper] No digital text found. Skipping Fast Lane -> FORCING Robust Pipeline.")
            attempt_fast_lane = False

    # --- TIER 2: FAST LANE OCR ---
    if attempt_fast_lane:
        print("   >>> Attempting Tier 2: Fast Lane Visual OCR...")

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

        if is_page_dirty(img_gray):
            print("   >>> [Fast Lane] Noise detected. Applying NL Means Denoising...")
            img_bgr = clean_digital_noise(img_bgr)

        try:
            data_fast = pt.image_to_data(img_bgr, config='--psm 3', output_type=pt.Output.DICT)
            raw_fast = pt.image_to_string(img_bgr, config='--psm 3')
            fast_text = basic_text_cleanup(raw_fast)

            confs = [int(c) for c in data_fast['conf'] if c != '-1']
            avg_conf = sum(confs) / len(confs) if confs else 0
            print(f"   >>> [Fast Lane] Confidence: {avg_conf:.2f}%")
        except Exception as e:
            print(f"   >>> [Fast Lane] OCR Error: {e}")
            fast_text = ""
            avg_conf = 0

        if len(fast_text.strip()) > 50 and not is_garbage_text(fast_text) and avg_conf > 50:
            print("   >>> SUCCESS: Fast Lane OCR extracted text.")

            page_text_to_add = f"--- PAGE {page_num + 1} (TIER 2: FAST LANE) ---\n\n{fast_text}\n" + "=" * 50 + "\n"
            score, status = get_page_quality_score(page_text_to_add, avg_conf)
            print(f"   >>> [Quality Gate] Score: {score:.2f} -> {status}")

            if "GARBAGE" not in status:
                # Write output immediately to disk for this page
                with open(page_md_path, "w", encoding="utf-8") as f:
                    f.write(page_text_to_add)

                cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 80), (0, 200, 0), -1)
                cv2.putText(img_bgr, "TIER 2: FAST LANE OCR", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                debug_images_list.append(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))

                continue  # <--- Skip Robust Lane
            else:
                print(f"   >>> Fast Lane Quality Gate Failed: {status}. FALLING THROUGH to Robust Lane...")
                if skew_summary and skew_summary[-1][0] == page_num + 1:
                    skew_summary.pop()
        else:
            print(f"   >>> Fast Lane failed (Conf: {avg_conf:.2f}% or Garbage). FALLING THROUGH to Robust Lane...")
            if skew_summary and skew_summary[-1][0] == page_num + 1:
                skew_summary.pop()

    # --- TIER 3: ROBUST PIPELINE ---
    print("   >>> FORCING Tier 3: Robust Pipeline.")

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

    is_clean_mode = False
    clean_text_psm3 = ""
    clean_conf_psm3 = 0.0
    data_psm3 = None

    if is_dirty:
        print("      [Robust] Mode: DIRTY -> Using Dual Pipeline")

        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        blur_layout = cv2.bilateralFilter(img_gray, 7, 55, 55)
        thresh_layout_base = cv2.adaptiveThreshold(blur_layout, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                   15, 15)
        thresh_layout = clean_heavy_grain(thresh_layout_base, min_area=50)

        blur_ocr = cv2.bilateralFilter(img_gray, 5, 25, 25)
        thresh_ocr = cv2.adaptiveThreshold(blur_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 12)
        healed_ocr_img = cv2.erode(thresh_ocr, np.ones((2, 2), np.uint8), iterations=1)

    else:
        print("      [Robust] Mode: CLEAN -> Decoupled Pipeline")
        is_clean_mode = True

        gray_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        thresh_layout = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        healed_ocr_img = cv2.fastNlMeansDenoising(img_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

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

    if is_clean_mode:
        print("      [Robust Clean] Running Parallel OCR (PSM 3)...")
        try:
            data_psm3 = pt.image_to_data(healed_ocr_img, config="--psm 3", output_type=pt.Output.DICT)
            raw_psm3 = pt.image_to_string(healed_ocr_img, config="--psm 3")
            clean_text_psm3 = basic_text_cleanup(raw_psm3)
            clean_conf_psm3 = get_ocr_confidence(data_psm3)
            print(f"      [Robust Clean] PSM 3 Confidence: {clean_conf_psm3:.2f}")
        except:
            clean_conf_psm3 = 0.0

    # 6. Layout Analysis
    edges = cv2.Canny(thresh_layout, 50, 150)
    layout_map = cv2.bitwise_not(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
    blocks = get_sorted_text_blocks(layout_map, min_w_pixels, min_h_pixels, DPI)
    blocks = filter_giant_blocks(blocks, healed_ocr_img.shape[1], healed_ocr_img.shape[0])
    blocks = remove_nested_blocks(blocks)

    print(f"      [Robust] Found {len(blocks)} raw blocks.")

    # 7. Extraction & Confidence Comparison
    vis_image_bgr = cv2.cvtColor(healed_ocr_img, cv2.COLOR_GRAY2BGR)

    layout_text_full = ""
    layout_conf_sum = 0
    layout_conf_count = 0
    block_results = []

    manual_block_count = len(blocks)
    valid_green_blocks = 0
    forced_psm3 = False

    if is_clean_mode and manual_block_count > 50:
        print(
            f"      [Robust Clean] Manual Layout Fragmented ({manual_block_count} blocks > 50). Instantly switching to PSM 3.")
        forced_psm3 = True
    else:
        for i, (x, y, w, h) in enumerate(blocks):
            roi = healed_ocr_img[y:y + h, x:x + w]
            if is_dense_graphic(roi):
                block_results.append((i, x, y, w, h, "image", "", 0))
                continue

            try:
                if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                    text_box, conf_box = "", 0
                    continue

                data_box = pt.image_to_data(roi, config="--oem 3 --psm 6", output_type=pt.Output.DICT)
                raw_box = pt.image_to_string(roi, config="--oem 3 --psm 6")

                text_box = basic_text_cleanup(raw_box)
                conf_box = get_ocr_confidence(data_box)
            except Exception as e:
                print(f"      [!] Tesseract Error on block {i}: {e}")
                text_box, conf_box = "", 0

            if is_garbage_text(text_box):
                block_results.append((i, x, y, w, h, "diagram", "", 0))
            else:
                block_results.append((i, x, y, w, h, "text", text_box, conf_box))
                layout_text_full += text_box + "\n\n"
                layout_conf_sum += conf_box
                layout_conf_count += 1
                valid_green_blocks += 1

    layout_avg_conf = (layout_conf_sum / layout_conf_count) if layout_conf_count > 0 else 0.0

    if is_clean_mode:
        print(
            f"      [Robust Clean] Comparison -> PSM 3 Conf: {clean_conf_psm3:.2f} vs Layout Conf: {layout_avg_conf:.2f}")

    use_psm3 = False
    reason = ""

    if is_clean_mode:
        try:
            psm3_block_count = len(set(data_psm3['block_num'])) if data_psm3 else 1
        except:
            psm3_block_count = 1

        print(
            f"      [Robust Clean] Structure -> Valid Green Blocks: {valid_green_blocks} (Raw: {manual_block_count}) | PSM 3 Blocks: {psm3_block_count}")

        if forced_psm3:
            use_psm3, reason = True, "Manual Layout Fragmented (>50 blocks)"
        elif clean_conf_psm3 > (layout_avg_conf + 8):
            use_psm3, reason = True, "Higher Confidence (+8)"
        elif valid_green_blocks <= 3 and psm3_block_count > min(valid_green_blocks, 3):
            if clean_conf_psm3 > 50:
                use_psm3, reason = True, "Better Layout Detection (Multi-block vs Single)"
        elif manual_block_count > 5 and valid_green_blocks < 2:
            if clean_conf_psm3 > 60:
                use_psm3, reason = True, "Manual Layout mostly garbage/noise"

    if use_psm3:
        print(f"      [Robust Clean] SWITCHING to PSM 3 Output. Reason: {reason}")
        final_page_text = f"--- PAGE {page_num + 1} (TIER 3: ROBUST CLEAN - PSM 3) ---\n\n{clean_text_psm3}\n" + "=" * 50 + "\n"
        cv2.rectangle(vis_image_bgr, (0, 0), (vis_image_bgr.shape[1], 50), (0, 255, 0), -1)
        cv2.putText(vis_image_bgr, f"SELECTED: PSM 3 ({reason})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        if is_clean_mode: print("      [Robust Clean] Keeping Manual Layout Output.")
        final_page_text = f"--- PAGE {page_num + 1} (TIER 3: ROBUST PIPELINE) ---\n\n"

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

    final_page_conf = clean_conf_psm3 if use_psm3 else layout_avg_conf
    score, status = get_page_quality_score(final_page_text, final_page_conf)
    print(f"   >>> [Quality Gate] Score: {score:.2f} -> {status}")

    # Write robust output immediately to disk for this page regardless of gate
    # (LightOn will easily overwrite it later if it ends up in the fail list)
    with open(page_md_path, "w", encoding="utf-8") as f:
        f.write(final_page_text)

    if "GARBAGE" in status:
        pages_for_dots_ocr.append(page_num + 1)

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

# --- ROUTING SUMMARY REPORT ---
print("\n" + "=" * 50)
print("             EXTRACTION SUMMARY")
print("=" * 50)
print(f"Total Pages Processed: {len(doc)}")
print(f"Successful Extractions (Tiers 1-3): {len(doc) - len(pages_for_dots_ocr)}")
print(f"Pages Routed to AI Fallback (dots.ocr): {len(pages_for_dots_ocr)}")

# Retained the pages_for_dots_ocr output logic
if pages_for_dots_ocr:
    print("\n[!] The following pages failed the quality check and need fallback parsing:")
    print(f"    Pages: {pages_for_dots_ocr}")
    with open("pages_for_dots.txt", "w") as f:
        f.write(",".join(map(str, pages_for_dots_ocr)))
    print("    (Saved list to 'pages_for_dots.txt')")
else:
    print("\n[+] All pages passed the quality check! No pages routed to fallback.")
    with open("pages_for_dots.txt", "w") as f:
        f.write("")
    print("    (Cleared 'pages_for_dots.txt')")
print("=" * 50)

if debug_images_list:
    debug_images_list[0].save(
        OUTPUT_PDF_NAME, "PDF", resolution=100.0, save_all=True, append_images=debug_images_list[1:]
    )
    print(f">>> Saved visual debug PDF to: {OUTPUT_PDF_NAME}")

if debug_skew_list:
    debug_skew_list[0].save(
        OUTPUT_SKEW_PDF_NAME, "PDF", resolution=100.0, save_all=True, append_images=debug_skew_list[1:]
    )
    print(f">>> Saved SKEW DEBUG PDF to: {OUTPUT_SKEW_PDF_NAME}")