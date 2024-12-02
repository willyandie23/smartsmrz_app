from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/summarization', methods=['GET', 'POST'])
def summarization():
    return render_template('summarization.html')

@app.route('/evaluation', methods=['GET'])
def evaluation():
    return render_template('evaluation.html')


if __name__ == '__main__':
    app.run(debug=True)
