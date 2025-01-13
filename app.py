from flask import Flask, render_template, request, jsonify
import fitz
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import re
from rouge_score import rouge_scorer

app = Flask(__name__)

MODEL_NAME = "pegasus-finetuned"
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)

def split_text(text, chunk_size=512):
    """Bagi teks panjang menjadi potongan kecil (chunks)."""
    words = text.split()
    chunks = []
    while words:
        chunk = words[:chunk_size]
        chunks.append(" ".join(chunk))
        words = words[chunk_size:]
    return chunks

def clean_text(text):
    """Bersihkan teks dengan menghapus karakter yang tidak diperlukan."""
    text = re.sub(r'\s+', ' ', text)  # Hilangkan spasi berlebih
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Hapus karakter non-alfabetik kecuali tanda baca
    text = text.strip()  # Hapus spasi di awal dan akhir teks
    return text

def summarize_chunk(chunk, max_summary_length=64):
    """Ringkas satu chunk teks."""
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_summary_length, num_beams=5, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_text(text, chunk_size=512, max_summary_length=64):
    """Fungsi untuk merangkum teks panjang."""
    chunks = split_text(text, chunk_size)
    summaries = []
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)}")
        try:
            summary = summarize_chunk(chunk, max_summary_length)
            summaries.append(summary)
        except Exception as e:
            print(f"Error processing chunk {idx+1}: {e}")
    combined_summary = " ".join(summaries)
    return combined_summary

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/summarization', methods=['GET', 'POST'])
def summarization():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                # Baca teks dari PDF
                pdf_document = fitz.open(stream=file.read(), filetype="pdf")
                text = ""
                for page in pdf_document:
                    text += page.get_text()

                # Validasi teks kosong
                if not text.strip():
                    return render_template('summarization.html', summary="Error: PDF is empty or unreadable!")

                # Bersihkan teks
                cleaned_text = clean_text(text)

                # Ringkas teks panjang
                final_summary = summarize_text(cleaned_text)

                return render_template('summarization.html', summary=final_summary)

            except Exception as e:
                print(f"Error: {str(e)}")
                return render_template('summarization.html', summary=f"An error occurred: {str(e)}")
        else:
            return render_template('summarization.html', summary="No file uploaded!")
    return render_template('summarization.html')


@app.route('/evaluation', methods=['GET'])
def evaluation():
    return render_template('evaluation.html')

@app.route('/evaluate_abstraks', methods=['POST'])
def evaluate_abstraks():
    data = request.get_json()

    abstract_1 = data.get('abstract_1')
    abstract_2 = data.get('abstract_2')

    # Initialize Rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate ROUGE scores
    scores = scorer.score(abstract_1, abstract_2)

    # Extract F1 scores
    rouge_1_f1 = scores['rouge1'].fmeasure
    rouge_2_f1 = scores['rouge2'].fmeasure
    rouge_l_f1 = scores['rougeL'].fmeasure

    # Calculate dynamic rating based on F1 scores
    average_f1_score = (rouge_1_f1 + rouge_2_f1 + rouge_l_f1) / 3

    # Define Rating based on average F1 score
    if average_f1_score > 0.6:
        rating = 'Excellent'
        remarks = 'The abstracts are highly similar and well-formed.'
    elif average_f1_score > 0.4:
        rating = 'Good'
        remarks = 'The abstracts are somewhat similar with room for improvement.'
    else:
        rating = 'Poor'
        remarks = 'The abstracts are quite different and need further refinement.'

    # Prepare response
    evaluation_result = {
        'rouge_1': {
            'precision': f"{scores['rouge1'].precision:.4f}",
            'recall': f"{scores['rouge1'].recall:.4f}",
            'fmeasure': f"{rouge_1_f1:.4f}",
        },
        'rouge_2': {
            'precision': f"{scores['rouge2'].precision:.4f}",
            'recall': f"{scores['rouge2'].recall:.4f}",
            'fmeasure': f"{rouge_2_f1:.4f}",
        },
        'rouge_l': {
            'precision': f"{scores['rougeL'].precision:.4f}",
            'recall': f"{scores['rougeL'].recall:.4f}",
            'fmeasure': f"{rouge_l_f1:.4f}",
        },
        'rating': rating,
        'remarks': remarks
    }

    return jsonify(evaluation_result)

if __name__ == '__main__':
    app.run(debug=True)
