import numpy as np                              # Thư viện làm việc với ma trận
import pandas as pd                             # Thư viện thao tác với dữ liệu
import matplotlib.pyplot as plt                 #Thư viện dùng để vẽ đồ thị
from sklearn import linear_model, datasets      # sklearn: thư viện hỗ trợ học máy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("dataBTL.csv", encoding="utf-8",sep=";") # Đọc dữ liệu từ một file, encoding file đầu vào là utf-8, các cột được ngăn bởi dấu ;
                                                         # kết quả trả về là một dataframe
one = np.ones((data.shape[0],1))            # Tạo một ma trận có  data.shape[0] = 150 hàng, ,hàng 1 cột, giá trị mỗi phần tử là 1 
data.insert(loc=0, column="B", value=one)   # Thêm một cột vào data tại vị trí cột là 0, tên cột là A, giá trị cột là ma trận one
data_X = data[["B", "Acreage"]]              # data_X là ma trận dữ liệu đầu vào (data sets) = tất cả dữ liệu của 2 cột A và Price có trong dataframe data 
data_y = data["Price"]                     # data_y là Vector của outcome = tất cả dữ liệu của cột Acreage có trong dataframe data 

#tách training và test sets
X_train,X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 50)   # Hàm train_test_split() tách tập dữ liệu mẫu thành 2 phần train sets và test sets
                                                                                    # với các điểm dữ liệu là ngẫu nhiên, test sets có size = 50 => train sets = 100
#fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept = False) #fit_intercept = False không tính toán các điểm cắt với truc y(Price)
regr.fit(X_train, y_train)                  # Hàm fit(X_train, y_train) dùng để training từ data train sets và label train sets để cho ra hàm dự đoán
Y_pred = regr.predict(X_test)               # Kiểm thử: predict(X_test) hàm dự đoán với tham số đầu vào là data test sets (X_test) trả về tập nhãn(lable sets) dự đoán được gán vào y_pred

plt.plot(data.Acreage, data.Price, "ro")        # Biểu diễn các điểm dữ liệu trong dataframe dựa vào Price và Acreage bằng các dấu chấm đỏ (ro) và được nối liền với nhau (-)
plt.plot(X_test.Acreage, Y_pred, color="black")  # Vẽ một đường thẳng màu đen biểu diễn các điểm theo các cặp tọa độ là giá trị Height trong X_test và Weight dự đoán tương ứng
                                                    #Giá trị của X_test hiện tại chứa cả cột B với value = 1
w_0 = regr.coef_[0]
w_1 = regr.coef_[1]
print (w_0,w_1) 

x0 = np.linspace(10,60,2)     # np.linspace(10,60,2) tạo ra 2 mẫu cho x0 là 10 và 60
y0 = w_0 + w_1*x0               # Hàm hồi quy tuyến tính của Weight theo Height(x0)
plt.plot(x0,y0, color = "yellow")    # vẽ một đường thẳng bỏi 2 điểm (x0=10, y0(10)) và (x0 = 60, y0(60)) màu vàng
plt.xlabel("Acreage")
plt.ylabel("Price")
plt.title("Biểu đồ hồi quy tuyến tính biểu diễn giá phòng trọ theo diện tích")
#plt.show()      # Hiển thị đồ thị trong cửa sổ Figure
print("Mời bạn nhập diện tích phòng muốn thuê: ")
s=0
s=float(input())
p=w_1*s + w_0
print("Giá phòng dự đoán với areage",s,"là: ",p)