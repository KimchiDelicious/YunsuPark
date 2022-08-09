## 라이브러리를 불러오고, 이후에 pkl 파일을 불러오기
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

print(os.getcwd() )

model = pickle.load(open('./ml_model/iris_base.pkl', 'rb'))
print(model)

### 플라스크 프로그램을 구동시키기 
app = Flask(__name__)

# 플라스크 앱의 루트 디렉터리를 초기화
# 플라스크 앱의 루트 디렉터리를 초기화
@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST']) 
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1,data2,data3,data4]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)