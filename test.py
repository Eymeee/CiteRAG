from app.utils.pdf_loader import load_pdf_pages
from app.utils.text_cleaning import clean_pages
from app.utils.chunking import chunk_pages, ChunkOptions

pdf_path = "tests/sample.pdf"


# 1 - Load
pages = load_pdf_pages(pdf_path, normalize_whitespace=True)

# 2 - Clean
cleaned_pages = clean_pages(pages)

# 3 - Chunk
opt = ChunkOptions() # default
chunks = chunk_pages(cleaned_pages, options=opt, doc_id="sample.pdf")

print(f"Loaded pages: {len(pages)}")
print(f"Chunks produced: {len(chunks)}")


# previews
if cleaned_pages:
    print("\n--- Page 1 cleaned preview (first 200 chars) ---")
    print(cleaned_pages[0].text[:200])

if chunks:
    c0 = chunks[0]
    print("--- Chunk 0 metadata ---")
    print(f"id={c0.id}")
    print(f"doc_id={c0.doc_id}")
    print(f"page_start={c0.page_start}, page_end={c0.page_end}")
    print(f"chunk_index={c0.chunk_index}")
    print("\n--- Chunk 0 preview (first 300 chars) ---")
    print(c0.text[:300])
