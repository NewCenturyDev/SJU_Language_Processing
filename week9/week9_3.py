from urllib.request import urlopen, Request

import pandas as pd
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm_notebook

DATA_OUT_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week9/result/"

# 미리 크롤링해 저장한 CSV를 읽어옴
df = pd.read_csv(DATA_OUT_PATH + 'best_sandwiches_list_chicago.csv', index_col=0)
df.head()

print(df['URL'][0])


def print_single_request():
    req = Request(
        df['URL'][0],
        headers={'User-Agent': 'Mozilla/5.0'}
    )

    # URL 오픈
    html = urlopen(req)
    soup_tmp = BeautifulSoup(html, 'html.parser')
    print(soup_tmp)

    # 첫번째 p 태그의 class = 'addy'를 찾는다.
    print(soup_tmp.find('p', 'addy'))

    # 첫번째 p 태그의 class = 'addy'를 찾고 text를 추출
    price_tmp = soup_tmp.find('p', 'addy').get_text()
    print(price_tmp)
    price_tmp.split()
    print(price_tmp.split()[0])
    print(price_tmp.split()[0][:-1])
    ''.join(price_tmp.split()[1:-2])


def print_three_requests():
    price = []
    address = []

    for n in df.index[:3]:
        req = Request(df['URL'][n], headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req)
        soup_tmp = BeautifulSoup(html, 'lxml')

        # p 태그의 class='addy에 있는 text를 추출
        gettings = soup_tmp.find('p', 'addy').get_text()

        price.append(gettings.split()[0][:-1])
        address.append(''.join(gettings.split()[1:-2]))
    # 수집 결과 확인
    print(price)
    print(address)


def print_all_requests():
    price = []
    address = []

    for n in tqdm_notebook(df.index):
        req = Request(df['URL'][n], headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req)
        soup_tmp = BeautifulSoup(html, 'lxml')

        # p 태그의 class='addy에 있는 text를 추출
        gettings = soup_tmp.find('p', 'addy').get_text()

        price.append(gettings.split()[0][:-1])
        address.append(''.join(gettings.split()[1:-2]))

    # 수집 결과 확인
    print(price)
    print(address)

    # 데이터 프레임에 신규 컬럼 추가
    df['Price'] = price
    df['Address'] = address

    # 컬럼 정렬
    result_df = df.loc[:, ['Rank', 'Cafe', 'Menu', 'Price', 'Address']]

    # 데이터 프레임의 인덱스를 'Rank'로 변경
    result_df.set_index('Rank', inplace=True)
    print(result_df.head())

    return result_df


print_single_request()
print_three_requests()

# 처리된 데이터
new_df = print_all_requests()
new_df.to_csv(DATA_OUT_PATH + 'best_sandwiches_list_chicago2.csv', sep=',', encoding='UTF-8')
