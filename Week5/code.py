import pickle
import numpy as np
from flask import request,Flask


app=Flask(__name__)

@app.route('/')
def home():
    return "Hello"
@app.route('/predict/',methods=['POST'])
def predict():
    model=pickle.load(open('iris.pkl','rb'))
    x = np.array([request.args.get("a"),
                 request.args.get("b"),
                 request.args.get("c"),
                 request.args.get("d")]).reshape((-1,4))
    
    return f"Answer{model.predict(x)}"

app.run()