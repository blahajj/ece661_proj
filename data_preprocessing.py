from pypdf import PdfReader
def preprocess_data(uploaded_file):
    # Read the file and preprocess text
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
    else:
        raise ValueError("Unsupported file type!")
    return text
