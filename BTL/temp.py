import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('DLNS-ALL.csv')

# phân tích số lượng nhân sự tuyển dụng qua các năm ở từng địa phương
provinces = ['Hai Duong','Hung Yen','Ha Noi','Ha Giang','Bac Ninh','Thai Binh','Dien Bien','Khac']
counters = np.empty(len(provinces))
counters.fill(0)

# lấy dữ liệu cần thiết
employee_df = data.loc[1:, ['Tỉnh']]
for index in range(0, employee_df.shape[0]):
    province = data.iloc[index]['Tỉnh']
    if province in provinces:
        counter_idx = provinces.index(province)
        counters[counter_idx] += 1
    else:
        counter_idx = provinces.index('Khac')
        counters[counter_idx] += 1

print(counters)
plt.title('Tỷ lệ vùng miền của nhân sự trong công ty ABC')
plt.pie(counters, labels= provinces, startangle= 0, autopct='%1.2f%%')
plt.axis('equal')
plt.legend(bbox_to_anchor=(1, 0, 0.5, 1))

plt.show()