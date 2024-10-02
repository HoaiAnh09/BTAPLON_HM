import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# 1. Đọc và tiền xử lý dữ liệu
df = pd.read_csv('student-mat.csv', sep=';')

# Chỉ lấy các cột cần thiết (bỏ traveltime)
df = df[['sex', 'studytime', 'failures', 'G3']]

# Biến đổi cột 'sex' thành nhãn số (Label Encoding)
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])

# Tách biến đầu vào và biến mục tiêu
X = df[['sex', 'studytime', 'failures']]
y = df['G3']

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Xây dựng các mô hình
# 2.1 Hồi quy tuyến tính
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)

# 2.2 Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
r2_lasso = r2_score(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)

# 2.3 Neural Network - MLPRegressor
mlp_model = MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
r2_mlp = r2_score(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        sex = request.form['sex']
        studytime = float(request.form['studytime'])
        failures = float(request.form['failures'])

        # Chuyển đổi giới tính (sex)
        sex = le.transform([sex])[0]
        
        # Chuẩn bị dữ liệu để dự đoán
        features = np.array([[sex, studytime, failures]])
        
        # Dự đoán với các mô hình
        pred_linear = linear_model.predict(features)[0]
        pred_lasso = lasso_model.predict(features)[0]
        pred_mlp = mlp_model.predict(features)[0]
        
        # Kết quả dự đoán và độ tin cậy cho từng mô hình
        return render_template('index.html', 
                               pred_linear=f'{pred_linear:.2f}', r2_linear=f'{r2_linear:.2f}', mse_linear=f'{mse_linear:.2f}', rmse_linear=f'{rmse_linear:.2f}',
                               pred_lasso=f'{pred_lasso:.2f}', r2_lasso=f'{r2_lasso:.2f}', mse_lasso=f'{mse_lasso:.2f}', rmse_lasso=f'{rmse_lasso:.2f}',
                               pred_mlp=f'{pred_mlp:.2f}', r2_mlp=f'{r2_mlp:.2f}', mse_mlp=f'{mse_mlp:.2f}', rmse_mlp=f'{rmse_mlp:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
