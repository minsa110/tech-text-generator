# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./model/medium-tech"
ft_tokenizer = AutoTokenizer.from_pretrained(model_path)
ft_model = AutoModelForCausalLM.from_pretrained(model_path)

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('text-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    
    ft_input_ids = ft_tokenizer.encode(text, return_tensors='pt')
    output = ft_model.generate(ft_input_ids, attention_mask = torch.ones_like(ft_input_ids), pad_token_id=ft_tokenizer.eos_token_id,
                              max_length=100, do_sample=True)# num_beams=2, no_repeat_ngram_size=3, early_stopping=False)

    return ft_tokenizer.decode(output[0], skip_special_tokens=True)