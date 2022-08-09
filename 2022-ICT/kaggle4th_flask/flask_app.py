from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('./model/income_gb.pkl', 'rb'))
app = Flask(__name__)

# 플라스크 앱의 루트 디렉터리를 초기화
@app.route('/')
def main():
    return render_template('start.html')

# request.form['']을 사용하여 HTML 페이지에서 데이터를 가져온다. 
# model.predict()를 통해 클래스를 예측한다. 
# 예측값에 따라 어떤 텍스트와 이미지를 보낼지, after.html에 설정.
@app.route('/predict', methods=['POST']) 
def start():
    val1 = request.form['a']
    val2 = request.form['b']
    val3 = request.form['c']
    val4 = request.form['d']
    arr = np.array([[val1, val2, val3, val4]])
    pred = model.predict(arr)
    print("start pred ", pred)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)