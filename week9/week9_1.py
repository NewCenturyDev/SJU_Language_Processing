from bs4 import BeautifulSoup

page = open("test_first.html", "r").read()
soup = BeautifulSoup(page, "html.parser")
# 웹 페이지 전체를 보기 좋게 표출
print("웹 페이지 전체를 보기 좋게 표출")
print(soup.prettify())

# children: 한 단계 아래에 있는 태그를 보기 위한 함수
print("children: 한 단계 아래에 있는 태그를 보기 위한 함수")
print(list(soup.children))

# soup의 내용 안에 있는 html 태그에 접근하기 위한 코드
html = list(soup.children)[2]

# 본문이 나오는 body 부분만 추출
print("본문이 나오는 body 부분만 추출")
body = list(html.children)[3]
print(body)

# 특정 태그만 추출
body_tag = soup.body

# body 태그 아래의 태그만 추출
print("body 태그 아래의 태그만 추출")
print(list(body.children))

# body 태그 아래에 있는 태그 숫자를 세어준다
print("body 태그 아래에 있는 태그 숫자를 세어준다")
print(len(list(body.children)))

# find_all (요새는 이런 방법을 많이 쓴다 -> 특정 태그를 모두 가져온다)
print("find_all (요새는 이런 방법을 많이 쓴다 -> 특정 태그를 모두 가져온다)")
print(soup.find_all("p"))
print(soup.find_all("p", class_="outer_text"))
print(soup.find_all("p", id="first"))

# find 특정 태그를 찾아낸다 (맨 처음 것만)
print("find 특정 태그를 찾아낸다 (맨 처음 것만)")
print(soup.find("p"))

# 모든 p태그를 찾아서 그 안의 텍스트만을 출력
print("모든 p태그를 찾아서 그 안의 텍스트만을 출력")
for each_tag in soup.find_all("p"):
    print(each_tag.get_text())

# 링크(a) 태그 가져오기
print("링크(a) 태그 가져오기")
links = soup.find_all("a")
print(links)

for each in links:
    href = each['href']
    text = each.string
    print("{}->{}".format(text, href))
