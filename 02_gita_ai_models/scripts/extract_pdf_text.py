import argparse
from pathlib import Path


def extract_pdf_text(pdf_path, output_path):
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit("PyMuPDF/fitz is required for PDF extraction.") from exc

    source = Path(pdf_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(source))
    chunks = []
    for index in range(doc.page_count):
        text = doc.load_page(index).get_text("text") or ""
        chunks.append(f"\n\n--- PAGE {index + 1} ---\n{text}")

    output.write_text("\n".join(chunks), encoding="utf-8")
    return {"pages": doc.page_count, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description="Extract raw text from the Bhagavad Gita PDF.")
    parser.add_argument(
        "--pdf",
        default="source_pdfs/bhagavad-gita-as-it-is.pdf",
        help="PDF file to extract.",
    )
    parser.add_argument(
        "--output",
        default="data/extracted_raw_text.txt",
        help="Raw text output path.",
    )
    args = parser.parse_args()
    result = extract_pdf_text(args.pdf, args.output)
    print(f"Extracted {result['pages']} pages to {result['output']}")


if __name__ == "__main__":
    main()

