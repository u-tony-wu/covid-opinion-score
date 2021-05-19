from flask import Flask, render_template, url_for, request, redirect
import re
import torch
import torch.nn as nn
import transformers

# Load trained ML models
# Part One - load text tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Part Two - generate Tweet embedding
distilBERT_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
distilBERT_model.eval()

# Part Three - generate opinion score
class TwoLayerScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 1)

    def forward(self, x):
        ret = self.relu(self.linear1(x))
        ret = self.linear2(ret)
        return torch.tanh(ret)

score_model = TwoLayerScoreNet()
score_model.load_state_dict(torch.load('models/TwoLayerScoreModel.pth'))

# Tweet text to opinion score
def make_prediction(sentence):
    sentence = re.sub(r'http\S+', '', sentence.lower())
    x = torch.tensor(tokenizer.encode(sentence)).reshape(1,-1)
    with torch.no_grad():
        cls = distilBERT_model(x).last_hidden_state[0][0].reshape(1,768)
    pred = score_model(cls)
    return pred.item()

exps = ["very negative",
        "somewhat negative",
        "basically neutral or irrelevant",
        "somewhat positive",
        "very positive"]

# App starts here
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/result', methods=['POST'])
def predict_score():
    if request.method == 'POST':
        input_data = request.form['tweet']
        input_data = input_data.strip()
        if input_data == "":
            return render_template('result.html', tweet=input_data, output=0, explanation="")
        score = round(make_prediction(input_data), 4)

        if score <= -0.6:
            explanation = exps[0]
        elif score <= -0.2:
            explanation = exps[1]
        elif score <= 0.2:
            explanation = exps[2]
        elif score <=0.6:
            explanation = exps[3]
        else:
            explanation = exps[4]

        return render_template('result.html', tweet=input_data, output=score, explanation=explanation)


    return redirect('/')


if __name__ == '__main__':
    app.run()
