# -*- coding: utf-8 -*-
"""
Created on Sat May 16 22:13:24 2020

@author: jquin
"""

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from fastai import *
from fastai.text import *

# Load text classifier model and encoder
path = "C:/Users/jquin/Desktop/Projects/Twitts/"
model = load_learner(path, 'final_model.pkl')

app = Flask(__name__)

def classify(document):
    label = {0: "Depressed", 1: "Normal"}
    pred = model.predict(str(document))
    pred_class = int(pred[0])
    proba = np.max([float(i) for i in pred[2]]) * 100
    return label[pred_class], proba

class ReviewForm(Form):
    message = TextAreaField("How are you feeling today?",[validators.DataRequired(),validators.length(min=3)])
    @app.route('/')
    def index():
        form = ReviewForm(request.form)
        return render_template('messageform.html', form=form)
    
    @app.route('/results', methods=['POST'])
    def results():
        form = ReviewForm(request.form)
        if request.method == 'POST' and form.validate():
            review = request.form["message"]
            y, proba = classify(review)
            return render_template('results.html',content=review,prediction=y,probability=round(proba, 2))
        return render_template('messageform.html', form=form)
    
if __name__ == '__main__':
        app.run(debug=True)