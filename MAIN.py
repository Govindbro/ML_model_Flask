from flask import Flask, render_template, request
import pickle
import numpy as np
# for loading purpose
model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)#it is basically returns name of a python script


#current base directory 127.0.0.1.5000
@app.route('/')
def man():
    #to get return whole html template
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    prad = model.predict(arr)
    return render_template('click.html', data=prad)


if __name__ == "__main__":
    app.run(debug=True)# we need not to re-run this if we done any changes again and again















