import os
import re
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader

# --- CONFIG ---
BASE_DIR = r"C:\Projects\InsChat\Export\your_instagram_activity\messages\inbox\mrproton_17842677620922366"
OUTPUT_DIR = r"C:\Projects\InsChat\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TXT_FILE = os.path.join(OUTPUT_DIR, "chat_converted.txt")
PDF_FILE = os.path.join(OUTPUT_DIR, "chat_diary.pdf")

def convert_pst_to_ist(ts_raw):
    """
    Converts 'Aug 07, 2025 10:49 pm' (PST, UTC-8) to IST (UTC+5:30), ignoring DST.
    Adjusts date as needed!
    """
    for fmt in ["%b %d, %Y %I:%M %p", "%b %d, %Y %I:%M:%S %p"]:
        try:
            dt_naive = datetime.strptime(ts_raw, fmt)
            pst_dt = dt_naive.replace(tzinfo=timezone(timedelta(hours=-8)))  # Always PST!
            ist_offset = timedelta(hours=13, minutes=30)
            dt_ist = pst_dt + ist_offset
            return dt_ist.strftime("%Y-%m-%d %H:%M:%S IST")
        except Exception:
            continue
    return ts_raw

def parse_html(path):
    """Extracts messages and timestamps from one HTML file."""
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    entries = []
    for msg in soup.find_all("div", class_="pam"):
        sender_tag = msg.find("h2", class_="_3-95 _2pim _a6-h _a6-i")
        sender = sender_tag.get_text(strip=True) if sender_tag else "Unknown"

        text = ""
        msg_body = msg.find("div", class_="_3-95 _a6-p")
        if msg_body:
            inner_divs = msg_body.find_all("div")
            for d in inner_divs:
                t = d.get_text(strip=True)
                if t:
                    text = t
                    break

        time_tag = msg.find("div", class_="_3-94 _a6-o")
        ts_raw = time_tag.get_text(strip=True) if time_tag else None
        ts_fmt = convert_pst_to_ist(ts_raw) if ts_raw else None

        photo_tag = msg.find("img")
        photo_src = photo_tag.get("src") if photo_tag and photo_tag.has_attr("src") else None

        entries.append({
            "sender": sender,
            "timestamp": ts_fmt,
            "text": text,
            "photo": photo_src
        })

    return entries

def get_true_image_path(photo_src):
    """Find the absolute path for an attached image, fixing relative path issues."""
    if not photo_src:
        return None
    # If the src is absolute, just check directly
    if os.path.isabs(photo_src) and os.path.exists(photo_src):
        return photo_src
    candidate1 = os.path.normpath(os.path.join(BASE_DIR, photo_src))
    if os.path.exists(candidate1):
        return candidate1
    candidate2 = os.path.normpath(os.path.join(BASE_DIR, "photos", os.path.basename(photo_src)))
    if os.path.exists(candidate2):
        return candidate2
    candidate3 = os.path.normpath(os.path.join(BASE_DIR, "media", os.path.basename(photo_src)))
    if os.path.exists(candidate3):
        return candidate3
    candidate4 = os.path.join(BASE_DIR, os.path.basename(photo_src))
    if os.path.exists(candidate4):
        return candidate4
    return None

def make_txt(entries, txt_path):
    """Creates the text version of the chat."""
    with open(txt_path, "w", encoding="utf-8") as f:
        for e in entries:
            if not e["text"] and not e["photo"]:
                continue
            f.write(f"[{e['timestamp']}] {e['sender']}: {e['text']}\n")
            if e["photo"]:
                f.write(f"[File: {e['photo']}]\n")
            f.write("\n")

def make_pdf(entries, pdf_path):
    """
    Creates a clean diary-style PDF with aspect-correct, non-split images.
    Maintains image ratios, keeps chat and image together on same page.
    """
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    chat_style = ParagraphStyle(
        name="Chat",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        spaceAfter=6,
        leading=14
    )
    img_max_dim = 400  # px, max width/height

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    story = []
    for idx, e in enumerate(entries, 1):
        if not e["text"] and not e["photo"]:
            continue
        block = []
        para = f"<b>{e['sender']}</b> <font size=8>({e['timestamp']})</font>"
        block.append(Paragraph(para, chat_style))
        if e["text"]:
            msgtext = e["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            block.append(Paragraph(msgtext, styles["BodyText"]))
        if e["photo"]:
            ext = os.path.splitext(e["photo"])[-1].lower()
            true_img_path = get_true_image_path(e["photo"])
            if ext in IMAGE_EXTENSIONS and true_img_path and os.path.exists(true_img_path):
                try:
                    img_reader = ImageReader(true_img_path)
                    iw, ih = img_reader.getSize()
                    scale = min(img_max_dim/iw, img_max_dim/ih, 1)
                    img_width, img_height = int(iw*scale), int(ih*scale)
                    img_obj = Image(true_img_path, width=img_width, height=img_height)
                    block.append(img_obj)
                except Exception:
                    block.append(Paragraph(f"[Image load error: {e['photo']}]", styles["Italic"]))
            else:
                block.append(Paragraph(f"[File: {e['photo']}]", styles["Italic"]))
        block.append(Spacer(1, 12))
        story.append(KeepTogether(block))
    if not story:
        story.append(Paragraph("⚠️ No messages detected — please confirm file path or HTML structure.", styles["Title"]))
    else:
        story.append(Paragraph(f"<b>Total messages parsed: {len(entries)}</b>", styles["Normal"]))
    doc.build(story)

def natural_sort_key(filename):
    """Ensures message_2.html comes before message_10.html."""
    match = re.search(r"message_(\d+)\.html", filename)
    return int(match.group(1)) if match else 0

if __name__ == "__main__":
    all_entries = []

    html_files = [f for f in os.listdir(BASE_DIR) if f.endswith(".html")]
    html_files = sorted(html_files, key=natural_sort_key)

    print(f"Found {len(html_files)} HTML files. Parsing chronologically...\n")

    for idx, file in enumerate(html_files, 1):
        path = os.path.join(BASE_DIR, file)
        print(f"📖 Parsing {file} ({idx}/{len(html_files)}) ...")
        entries = parse_html(path)
        all_entries.extend(entries)

    print(f"\nTotal messages parsed: {len(all_entries)}")
    print("📝 Creating text file...")
    make_txt(all_entries, TXT_FILE)

    print("📘 Creating PDF (this might take a while)...")
    make_pdf(all_entries, PDF_FILE)

    print("\n✅ Done! Files saved in:")
    print(f"   TXT: {TXT_FILE}")
    print(f"   PDF: {PDF_FILE}")
