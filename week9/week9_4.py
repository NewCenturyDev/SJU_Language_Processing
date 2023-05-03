from urllib.request import urlopen, Request

import pandas as pd
from bs4 import BeautifulSoup

DATA_OUT_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week9/result/"
NAVER_NEWS_LIST_URL = "https://news.naver.com/main/clusterArticles.naver?id=c_202305031120_00000012&mode=LSD&mid=shm" \
                      "&sid1=101&oid=629&aid=0000215015 "

# 미리 크롤링해 저장한 CSV를 읽어옴
df = pd.read_csv(DATA_OUT_PATH + 'best_sandwiches_list_chicago.csv', index_col=0)
df.head()


def crawl_news_list():
    req = Request(NAVER_NEWS_LIST_URL, headers={'User-Agent': 'Mozilla/5.0'})

    # URL 오픈
    html = urlopen(req)
    soup_tmp = BeautifulSoup(html, 'html.parser')

    # 뉴스 기사 링크 가져오기
    news_photos = soup_tmp.find_all('dt', 'photo')
    news_links = []
    for p in news_photos:
        news_links.append(p.a['href'])

    return news_links


def crawl_news_single(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(req)
    soup_tmp = BeautifulSoup(html, 'lxml')

    # p 태그의 class='addy에 있는 text를 추출
    return {
        'title': soup_tmp.find('h2', id='title_area').span.get_text(),
        'timestamp': soup_tmp.find('span', 'media_end_head_info_datestamp_time _ARTICLE_DATE_TIME').get_text(),
        'reporter': soup_tmp.find('em', 'media_end_head_journalist_name').get_text()
    }


links = crawl_news_list()
title = []
timestamp = []
reporter = []

for link in links:
    result = crawl_news_single(link)
    title.append(result['title'])
    timestamp.append(result['timestamp'])
    reporter.append(result['reporter'])

# 데이터프레임으로 변환 및 파일 저장
df = pd.DataFrame(list(zip(title, timestamp, reporter)), columns=['title', 'timestamp', 'reporter'])
print(df.head())
df.to_csv(DATA_OUT_PATH + 'naver_news.csv', sep=',', encoding='UTF-8')
