from flask import Flask, render_template, request, redirect, url_for
from transformers import MarianMTModel, MarianTokenizer
import pdfplumber
import os

app = Flask(__name__)

# Load the pre-trained MarianMT model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-zh-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_chinese_to_english(text):
    """ Translates Chinese text to English. """
    try:
        tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
        translated = model.generate(**tokenized_text)
        translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return translated_text[0]
    except Exception as e:
        return f"Error during translation: {e}"

def extract_text_from_pdf(pdf_path):
    """ Extracts text from a PDF file. """
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
    return "\n".join(extracted_text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded PDF to a temporary location
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Extract text and translate
        pdf_text = extract_text_from_pdf(file_path)
        if pdf_text:
            translated_text = translate_chinese_to_english(pdf_text)
            return render_template('result.html', translated_text=translated_text)
        else:
            return "No text extracted from the PDF."

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
