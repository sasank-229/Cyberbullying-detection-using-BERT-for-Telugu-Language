from flask import Flask, request, render_template,url_for
from tqdm import tqdm
import numpy as np
# import nbformat
# from nbconvert import PythonExporter
# import os
import torch
from transformers import AutoModel,AutoTokenizer
import pickle
from xgboost import XGBClassifier

app = Flask(__name__)

# Load the model during the application startup
# @before_first_request
# def load_model():
#     try:
#         with open('static/ipynbFiles/classifier2.pkl', 'rb') as file:
#             current_app.clf = pickle.load(file)
#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         abort(500)  # Internal Server Error
# app.before_first_request(load_model)

def model_extract(input_string):
    param ={'maxLen' :256,}
    model = AutoModel.from_pretrained("ai4bharat/indic-bert")
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

    def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0):
        padded_sequences = []
        for seq in sequences:
            if padding == 'pre':
                padded_seq = np.pad(seq, (maxlen - len(seq), 0), 'constant', constant_values=value)
            elif padding == 'post':
                padded_seq = np.pad(seq, (0, maxlen - len(seq)), 'constant', constant_values=value)
            else:
                raise ValueError("Padding should be 'pre' or 'post'.")

            if truncating == 'pre':
                padded_seq = padded_seq[-maxlen:]
            elif truncating == 'post':
                padded_seq = padded_seq[:maxlen]
            else:
                raise ValueError("Truncating should be 'pre' or 'post'.")

            padded_sequences.append(padded_seq)

        return np.array(padded_sequences, dtype=dtype)


    def create_attention_masks(input_ids):
        attention_masks = []
        for seq in tqdm(input_ids):
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        return np.array(attention_masks)

    def getFeaturesandLabel(single_string, label):
        # Wrap the single string in a list
        sentences = ["[CLS] " + single_string + " [SEP]"]

        # Tokenize and preprocess
        tokenizer_texts = list(map(lambda t: tokenizer.tokenize(t)[:512], tqdm(sentences)))
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tqdm(tokenizer_texts)]

        # Pad sequences and create attention masks
        input_ids = pad_sequences(sequences=input_ids, maxlen=param['maxLen'], dtype='long', padding='post', truncating='post')
        attention_masks_data = create_attention_masks(input_ids)

        # Convert to torch tensors
        X_data = torch.tensor(input_ids)
        attention_masks_data = torch.tensor(attention_masks_data)
        y_data = torch.tensor(label)

        return X_data, attention_masks_data, y_data 
    
    text_input=input_string
    label_input = [0]
    X_data, attention_masks_data, y_data = getFeaturesandLabel(text_input, label_input)
    return X_data


# def model_heart():
#     # Path to the notebook file
#     notebook_path = os.path.join('static', 'ipynbFiles', 'trail.ipynb')
#     # Read the notebook content
#     with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
#         notebook_content = nbformat.read(notebook_file, as_version=4)
#     # Create a PythonExporter
#     python_exporter = PythonExporter()
#     # Convert the notebook to a Python script
#     python_script, _ = python_exporter.from_notebook_node(notebook_content)
#     print(python_script)
#     # Execute the Python script
#     exec(python_script)

# model_heart()
# Now you can use the variables and functions defined in the notebook in your app.py
from tempCodeRunnerFile import match
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' ,methods=['POST','GET'])
def predict():
    input_string=request.form['text']
    # # print('text: ',input_string)
    with open('static/ipynbFiles/classifier_10epochs_updated.pkl','rb') as file:
        clf=pickle.load(file)
    
    # if any(c in input_string for c in match):
    #     prediction = [0]
    # else:
    #     ans=model_extract(input_string)
    #     print('torch.tensor variable: ',ans)
    #     prediction = clf.predict(ans)

    # # print('prediction=',prediction)
    # if prediction==[0]:
    #     return render_template('index.html', pred='Cyberbullying Text', question='వాక్యం -   '+input_string)
    # else:
    #     return render_template('index.html', pred='Non-Cyberbullying Text', question='వాక్యం -   '+input_string)
    return render_template('index.html')

#for creating a pickle file: 
#   with open('classifier.pkl','wb') as file:
#       pickle.dump(xgb, file)
