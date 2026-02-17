from app.utils.pdf_loader import load_pdf_pages
from app.utils.text_cleaning import clean_pages
from dataclasses import replace

pdf_path = "tests/sample.pdf"

pages = load_pdf_pages(pdf_path, normalize_whitespace=True)
cleaned_pages = clean_pages(
    pages,
    get_text=lambda p: p.text,
    set_text=lambda p, new_text: replace(p, text=new_text),
)

print(f"Loaded {len(pages)} pages")
num_char = 2000
for page in cleaned_pages:
    preview = page.text[:num_char]
    print(f"\nPage {page.page_number} preview first {num_char} characters: \n{preview}...")
