Kế Hoạch Thực Thi Chi Tiết: Hệ Thống Chẩn Đoán Bệnh Tật
0. Chuẩn Bị Môi Trường & Cấu Trúc Dự Án
Bản chất
Tạo một không gian làm việc sạch sẽ, có tổ chức. Điều này giúp quản lý code, dữ liệu và các mô hình đã huấn luyện một cách dễ dàng, tránh nhầm lẫn.
Cách làm
Tạo cấu trúc thư mục:
code
Code
DU_AN_CHAN_DOAN_BENH/
├── data/
│   └── Disease and symptoms dataset.csv
├── notebooks/
│   └── 1_EDA_and_Preprocessing.ipynb
├── src/
│   └── train.py
│   └── app.py
├── models/
│   └── (Đây là nơi lưu các mô hình và encoder)
└── requirements.txt
Tạo môi trường ảo và cài đặt thư viện:
Mở terminal trong thư mục DU_AN_CHAN_DOAN_BENH.
Chạy python -m venv venv để tạo môi trường ảo.
Kích hoạt môi trường: source venv/bin/activate (trên macOS/Linux) hoặc .\venv\Scripts\activate (trên Windows).
Tạo file requirements.txt với nội dung:
code
Code
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
Chạy pip install -r requirements.txt để cài đặt.
Bước 1 & 2: Phân Tích và Tiền Xử Lý Dữ Liệu (Gộp chung)
Bản chất
Vì dữ liệu đã có cấu trúc, chúng ta có thể thực hiện cả hai bước này trong một file Jupyter Notebook để có sự tương tác và trực quan hóa ngay lập tức. Mục tiêu là hiểu dữ liệu và chuẩn bị nó sẵn sàng cho việc huấn luyện.
Cách làm (Trong file notebooks/1_EDA_and_Preprocessing.ipynb)
Load Dữ Liệu: Dùng pandas.read_csv() để tải Disease and symptoms dataset.csv.
Thực hiện EDA (Phân tích):
Xem lướt: Dùng df.head(), df.info(), df.shape để nắm thông tin cơ bản.
Phân tích các lớp (bệnh):
Đếm số bệnh duy nhất: df['diseases'].nunique().
Đếm số mẫu mỗi bệnh: df['diseases'].value_counts().
Trực quan hóa: Dùng seaborn.countplot() hoặc matplotlib.pyplot.bar() để vẽ biểu đồ 20 bệnh có nhiều mẫu nhất.
Phân tích các thuộc tính (triệu chứng):
Tính tổng mỗi cột triệu chứng: df.drop('diseases', axis=1).sum().sort_values(ascending=False).
Trực quan hóa: Vẽ biểu đồ 20 triệu chứng phổ biến nhất.
Thực hiện Preprocessing (Tiền xử lý):
Tách Features (X) và Target (y):
X = df.drop('diseases', axis=1)
y = df['diseases']
Mã hóa Nhãn:
Khởi tạo LabelEncoder từ sklearn.preprocessing.
y_encoded = label_encoder.fit_transform(y)
Lưu lại các thành phần quan trọng: Dùng joblib để lưu:
joblib.dump(label_encoder, '../models/label_encoder.joblib')
joblib.dump(X.columns.tolist(), '../models/feature_list.joblib') -> Cực kỳ quan trọng để dùng ở Bước 5.
Chia dữ liệu:
Dùng train_test_split với test_size=0.2, random_state=42, và stratify=y_encoded.
Lưu dữ liệu đã xử lý: Lưu X_train, X_test, y_train, y_test ra các file riêng biệt (ví dụ: .csv hoặc .npy) trong thư mục data/processed/ để Bước 3 có thể sử dụng.
Thông tin cần có cho báo cáo
Các bảng thống kê và biểu đồ đã tạo.
Nhận xét về độ mất cân bằng của dữ liệu.
Giải thích rõ ràng tại sao phải lưu label_encoder và feature_list.
Nêu rõ kích thước của các tập dữ liệu sau khi chia.
Câu hỏi giáo viên có thể hỏi
"Danh sách các features (feature_list) em lưu lại để làm gì?"
Trả lời: "Dạ, em lưu lại để đảm bảo rằng khi người dùng nhập triệu chứng ở Bước 5, em có thể tạo ra một vector đầu vào có thứ tự các triệu chứng hoàn toàn trùng khớp với thứ tự mà mô hình đã học. Nếu không có danh sách này, thứ tự có thể bị sai lệch và dẫn đến dự đoán sai hoàn toàn."
Bước 3: Huấn Luyện & Lựa Chọn Mô Hình
Bản chất
Tổ chức một "cuộc thi" giữa các thuật toán Machine Learning truyền thống để tìm ra mô hình hoạt động hiệu quả nhất trên dữ liệu của bạn trong thời gian ngắn nhất.
Cách làm (Trong file src/train.py)
Viết một hàm load_data() để đọc các file đã xử lý từ Bước 2.
Định nghĩa các mô hình: Tạo một dictionary chứa các thuật toán muốn thử: DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, MultinomialNB.
Tối ưu tốc độ: Với RandomForestClassifier và LogisticRegression, hãy thêm tham số n_jobs=-1 để tận dụng tất cả các nhân CPU.
Tạo vòng lặp huấn luyện và đánh giá:
Lặp qua dictionary các mô hình.
Trong mỗi vòng lặp, ghi lại thời gian bắt đầu, fit mô hình trên dữ liệu train, predict trên dữ liệu test, ghi lại thời gian kết thúc.
Tính toán các chỉ số: accuracy_score và classification_report(output_dict=True) để lấy precision/recall/f1-score trung bình.
In kết quả: In ra một bảng so sánh rõ ràng, sắp xếp theo accuracy giảm dần.
Chọn mô hình tốt nhất: Dựa vào bảng kết quả, chọn ra mô hình có sự cân bằng tốt nhất giữa accuracy và thời gian huấn luyện.
Thông tin cần có cho báo cáo
Bảng so sánh chi tiết hiệu suất các mô hình.
Lý giải sự lựa chọn mô hình tốt nhất (ví dụ: "Em chọn Random Forest vì nó cho accuracy cao nhất, mặc dù thời gian huấn luyện lâu hơn Logistic Regression một chút, nhưng sự chênh lệch hiệu suất là đáng kể.").
Câu hỏi giáo viên có thể hỏi
"Tại sao các mô hình học sâu như mạng nơ-ron không được đưa vào so sánh?"
Trả lời: "Dạ, vì dữ liệu đầu vào đã ở dạng thuộc tính có cấu trúc (structured feature matrix) chứ không phải văn bản thô. Với dạng dữ liệu này, các thuật toán Machine Learning truyền thống như Random Forest thường cho hiệu suất rất cạnh tranh, trong khi thời gian huấn luyện nhanh hơn và mô hình dễ giải thích hơn. Việc sử dụng mạng nơ-ron sẽ làm tăng độ phức tạp không cần thiết cho bài toán này."
Bước 4: Tinh Chỉnh Siêu Tham Số
Bản chất
Sau khi đã chọn được "vận động viên" tốt nhất (ví dụ: Random Forest), chúng ta sẽ "tinh chỉnh trang bị" cho họ để đạt hiệu suất tối ưu.
Cách làm (Trong file src/train.py, có thể thêm một hàm mới)
Lấy mô hình tốt nhất: Dựa trên kết quả ở Bước 3.
Định nghĩa không gian tìm kiếm: Tạo một dictionary param_dist chứa các siêu tham số muốn thử. Giữ cho nó nhỏ để chạy nhanh.
Ví dụ cho RandomForest: {'n_estimators': [100, 200], 'max_depth': [20, 30, None], 'min_samples_leaf': [1, 2]}.
Sử dụng RandomizedSearchCV:
Khởi tạo RandomizedSearchCV với mô hình, param_dist, n_iter=10 (thử 10 tổ hợp ngẫu nhiên), cv=3 (cross-validation 3 lần), n_jobs=-1.
fit đối tượng search này trên toàn bộ tập huấn luyện (X_train, y_train).
Lấy kết quả và lưu mô hình cuối cùng:
In ra bộ tham số tốt nhất: search.best_params_.
Lấy mô hình tốt nhất: best_model = search.best_estimator_.
Lưu lại mô hình cuối cùng: joblib.dump(best_model, '../models/final_model.joblib').
Thông tin cần có cho báo cáo
Giải thích các siêu tham số đã chọn để tinh chỉnh và ý nghĩa của chúng.
Trình bày bộ siêu tham số tốt nhất đã tìm thấy.
So sánh accuracy trên tập test của mô hình trước và sau khi tinh chỉnh.
Câu hỏi giáo viên có thể hỏi
"Cross-validation (tham số cv=3) trong quá trình tuning có ý nghĩa gì?"
Trả lời: "Dạ, nó giúp đánh giá mỗi tổ hợp siêu tham số một cách đáng tin cậy hơn. Thay vì chỉ kiểm tra trên một lần chia dữ liệu, nó sẽ chia tập huấn luyện thành 3 phần, huấn luyện trên 2 phần và kiểm tra trên phần còn lại, lặp lại 3 lần. Điểm số cuối cùng là trung bình của 3 lần đó, giúp kết quả ít bị ảnh hưởng bởi sự may rủi và đảm bảo bộ tham số được chọn thực sự tốt."
Bước 5: Xây Dựng Ứng Dụng Chẩn Đoán
Bản chất
Biến mô hình đã được huấn luyện và tinh chỉnh thành một công cụ thực tế mà người dùng có thể tương tác.
Cách làm (Trong file src/app.py)
Viết hàm load_artifacts(): Hàm này sẽ tải final_model.joblib, label_encoder.joblib, và feature_list.joblib từ thư mục models/.
Viết hàm predict_disease(symptoms_list):
Hàm này nhận đầu vào là một danh sách các chuỗi triệu chứng (ví dụ: ['headache', 'cough']).
Bước quan trọng: Tạo một vector input_vector gồm toàn số 0, có độ dài bằng len(feature_list).
Lặp qua symptoms_list:
Với mỗi triệu chứng, tìm index của nó trong feature_list.
Nếu tìm thấy, gán input_vector[index] = 1.
Dùng model.predict([input_vector]) để dự đoán (lưu ý [] để tạo thành mảng 2D).
Dùng label_encoder.inverse_transform() để dịch kết quả số về tên bệnh.
Trả về tên bệnh.
Tạo giao diện dòng lệnh:
Trong if __name__ == "__main__":, gọi hàm load_artifacts().
Tạo một vòng lặp while True để liên tục hỏi người dùng nhập triệu chứng.
Hiển thị một ví dụ về cách nhập: "Hãy nhập các triệu chứng, cách nhau bởi dấu phẩy (ví dụ: headache,cough,fever):"
Lấy chuỗi người dùng nhập, .strip().split(',') để tạo thành danh sách.
Gọi hàm predict_disease() và in kết quả chẩn đoán ra màn hình.
Thông tin cần có cho báo cáo
Sơ đồ khối (flowchart) mô tả cách ứng dụng hoạt động.
Ảnh chụp màn hình các ví dụ sử dụng, cho thấy người dùng nhập triệu chứng và nhận được kết quả.
Thảo luận về các hạn chế và hướng phát triển trong tương lai (ví dụ: giao diện web, thu thập thêm dữ liệu).
Câu hỏi giáo viên có thể hỏi
"Nếu người dùng nhập một triệu chứng không có trong feature_list thì sao?"
Trả lời: "Dạ, trong code của em, em có xử lý trường hợp này. Khi tìm index của triệu chứng trong feature_list, nếu không tìm thấy, em sẽ bỏ qua và có thể in ra một thông báo cho người dùng biết rằng triệu chứng đó không được hệ thống nhận dạng. Điều này đảm bảo ứng dụng không bị lỗi và chỉ đưa ra dự đoán dựa trên các thông tin mà nó đã được học."