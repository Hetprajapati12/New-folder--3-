import fitz  # PyMuPDF

# Load the PDF
pdf_path = "C:/Users/hetpr/OneDrive/Desktop/New folder (3)/policy-booklet-0923.pdf"
doc = fitz.open(pdf_path)

# Extract text from PDF
text_data = []
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text()
    text_data.append(text)

# Save extracted text to a file
with open("policy_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(text_data))
