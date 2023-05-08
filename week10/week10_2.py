import numpy as np
import pandas as pd
from glob import glob
import platform
import matplotlib
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns

# 백엔드 설정
matplotlib.use('TkAgg')

glob('./data/지역*.xls')
stations_files = glob('./data/지역*.xls')

tmp_raw = []

for file_name in stations_files:
    tmp = pd.read_excel(file_name, header=2)
    tmp_raw.append(tmp)

station_raw = pd.concat(tmp_raw)
station_raw.info()

print(station_raw.head())

# 필요한 데이터만 가져와서 전처리
stations = pd.DataFrame({
    "Oil_store": station_raw['상호'],
    "주소": station_raw['주소'],
    "가격": station_raw['휘발유'],
    "셀프": station_raw['셀프여부'],
    "상표": station_raw['상표'],
})

stations['구'] = [eachAddress.split()[1] for eachAddress in stations['주소']]

stations['가격'] = [float(value) for value in stations['가격']]

stations.reset_index(inplace=True)
del stations['index']

stations.info()

path = "c:/Windows/Fonts/malgun.ttf"


if platform.system == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print("Unknown system... sorry~~~~")

# 가격 정보에 대해 셀프 여부로 나누어 박스 플롯 그리기
stations.boxplot(column='가격', by='셀프', figsize=(12, 8))
plt.show()

# 가격 정보에 대해 상표로 나누어 박스 플롯 그리기
plt.figure(figsize=(12, 8))
sns.boxplot(x='상표', y='가격', hue='셀프', data=stations, palette='Set3')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='상표', y='가격', data=stations, palette='Set3')
sns.swarmplot(x='상표', y='가격', data=stations, color='.6')
plt.show()

# 구별 가격 정렬
print(stations.sort_values(by='가격', ascending=False).head(10))
print(stations.sort_values(by='가격', ascending=True).head(10))

gu_data = pd.pivot_table(stations, index=['구'], values=['가격'], aggfunc=np.mean)
gu_data.head()
