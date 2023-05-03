import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from urllib.parse import urljoin
import re


DATA_OUT_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week9/result/"


def crowl_finance_naver_marketindex():
    url = "https://finance.naver.com/marketindex/"
    page = urlopen(url)
    soup = BeautifulSoup(page, "html.parser")
    soup.prettify()
    print(soup.find_all("span", "value"))
    print(soup.find_all("span", "value")[0].string)


def crowl_sandwich_chicago():
    url_base = "http://www.chicagomag.com"
    # 하위 주소
    url_sub = "/Chicago-Magazine/November-2012/Best-Sandwiches-Chicago/"
    url = url_base + url_sub

    # 브라우저 에이전트 체크 방식의 크롤러 방지 시스템을 우회하기 위해 User-Agent 조작
    req = Request(
        url, headers={"User-Agent": "Mozilla/5.0"}
    )

    html = urlopen(req)
    soup = BeautifulSoup(html, "html.parser")
    print(soup)

    # 데이터 확인
    # class로 sammy를 가진 div를 추출해서 개수를 세리고 0번쨰 내용 확인
    divs = soup.find_all("div", "sammy")
    print("div 개수: {}".format(len(divs)))
    print(divs[0])

    # 각각의 div에서 sammyRank를 찾는다
    firstdiv = divs[0]
    firstdiv.find(class_="sammyRank").get_text()
    firstdiv.find(class_="sammyListing").get_text()
    links = firstdiv.find("a")["href"]
    print(links)

    tmp_string = firstdiv.find(class_="sammyListing").get_text()
    re.split("\n|\r\n", tmp_string)

    print(re.split("\n|\r\n", tmp_string)[0])
    print(re.split("\n|\r\n", tmp_string)[1])

    # 데이터 정리
    rank = []
    main_menu = []
    cafe_name = []
    url_add = []

    list_soup = soup.find_all("div", "sammy")

    # 찾아낸 각각의 div에 대하여 텍스트 및 랭크, href 링크를 추출
    for item in list_soup:
        rank.append(item.find(class_="sammyRank").get_text())
        tmp_string = item.find(class_="sammyListing").get_text()
        main_menu.append(re.split("\n|\r\n", tmp_string)[0])
        cafe_name.append(re.split("\n|\r\n", tmp_string)[1])
        url_add.append(urljoin(url_base, item.find("a")["href"]))

    # rank 리스트의 5번쨰까지 출력
    print(rank[:5])
    print(main_menu[:5])
    print(cafe_name[:5])
    print(url_add[:5])
    len(rank), len(main_menu), len(cafe_name), len(url_add)

    # 데이터 프레임으로 변환 및 칼럼 순서
    data = {"Rank": rank, "Menu": main_menu, "Cafe": cafe_name, "URL": url_add}
    df = pd.DataFrame(data, columns=["Rank", "Cafe", "Menu", "URL"])
    df.head(5)
    df.to_csv(DATA_OUT_PATH + "best_sandwiches_list_chicago.csv")


def crowl_finance_naver_main():
    url = "https://finance.naver.com/"
    page = urlopen(url)

    soup = BeautifulSoup(page, "html.parser")
    # HTML을 보기 좋게 정렬
    print(soup.prettify())

    # 특정 태그에 접근
    rates = soup.find_all("span", class_="num")
    print("시장 지수들 - 0번째 데이터는 쓰레기 값")
    print(rates)
    print(rates[0].string)
    del rates[0]
    heading_divs = soup.find_all("div", class_="heading_area")
    titles = []
    for div in heading_divs:
        titles.append(div.find_all("span", class_="blind")[0].get_text())

    # 주식 시장 지수를 종류별로 출력
    print("주식 시장 지수 크롤링 성공")
    for idx, market_rate in enumerate(rates):
        print("{} -> {}".format(titles[idx], market_rate.get_text()))


# crowl_finance_naver_marketindex()
crowl_sandwich_chicago()
# crowl_finance_naver_main()
