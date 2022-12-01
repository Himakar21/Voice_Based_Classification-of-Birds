import numpy as np
from flask import Flask, request, jsonify, render_template , flash,redirect, url_for

app = Flask(__name__)
import IPython.display as ipd
import librosa.display
from tensorflow.keras.models import load_model
model = load_model("C:\\Users\\LENOVO\\Desktop\\SE\\Project\\cnn_model1.hdf5")
labels = ['amewig','amewoo','amtspa','annhum','astfly','baisan','baleag','balori','banswa','barswa','bawwar','belkin1','Unknown']

# Given an numpy array of features, zero-pads each ocurrence to max_padding
def add_padding(features, mfcc_max_padding=174):
    padded = []
    # Add padding
    for i in range(len(features)):
        px = features[i]
        size = px[0]
        # Add padding if required
        if (size < mfcc_max_padding):
            xDiff = mfcc_max_padding - size
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            px = np.pad(px, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        
        padded.append(px)

    return padded

@app.route('/')
def hello_world():
    return render_template("login.html")
database={'himakar':'1234567','hemanth':'123','charan':'123','jasal':'123'}

@app.route('/form_login',methods=['POST','GET'])
def login():
    name1=request.form['username']
    pwd=request.form['password']
    if name1 not in database:
        return render_template('login.html',info='Invalid User')
    else:
        if database[name1]!=pwd:
            return render_template('login.html',info='Invalid Password')
        else:
	         return render_template('home.html',name="Welcome "+name1)

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    UPLOAD_FOLDER = 'C:/Users/LENOVO/Desktop/SE/Project/test'
    ALLOWED_EXTENSIONS = {'wav'}
    if request.method == 'POST':
        if 'bird_audio' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['bird_audio']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename != '':    
            y, sample_rate = librosa.load(UPLOAD_FOLDER+"/"+str(file.filename), res_type='kaiser_best',duration=10)
            normalized_y = librosa.util.normalize(y)
            mfcc = librosa.feature.mfcc(y=normalized_y, sr=sample_rate, n_mfcc=40)
            normalized_mfcc = librosa.util.normalize(mfcc)
            mfcc_max_padding = 0
            shape = normalized_mfcc.shape[1]
            if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
                    xDiff = mfcc_max_padding - shape
                    xLeft = xDiff//2
                    xRight = xDiff-xLeft
                    #normalized_mfcc = np.pad(normalized_mfcc,(xLeft, xRight), mode='constant')
            #padded_features = add_padding(normalized_mfcc,174)
            X = np.array(normalized_mfcc)
            X_test = X.reshape(1, 40, 431, 1)
            y_probs = model.predict(X_test)
            #yhat_probs = np.argmax(y_probs, axis=1)
            #y_true = np.sum(yhat_probs, axis=0)
            #print(y_probs,yhat_probs,y_trues)
            max=0
            max_index=-1
            print(y_probs,y_probs[0])
            y_probs=y_probs[0]
            for i in range(len(y_probs)):
                if (y_probs[i]>max):
                    max=y_probs[i]
                    max_index = i
            if(max>0.54):
                y_true = max_index
            else:
                y_true=12
            print("http://127.0.0.1:8887/test/"+str(file.filename))
            return render_template('home.html', prediction=str("Prediction by BirderAI :")+labels[y_true].upper(),imgsrc="http://127.0.0.1:8887/public/"+labels[y_true].upper()+".jpg",audiosrc="http://127.0.0.1:8887/test/"+str(file.filename))


if __name__ == "__main__":
    app.run(debug=True)
