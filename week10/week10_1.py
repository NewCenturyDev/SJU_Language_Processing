from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from tqdm import tqdm_notebook


# 네이버 스크린샷 해보기
def naver_screenshot():
    # selenium을 사용하여 크롬창을 연다
    # 이때, 크롬드라이버가 필요하다.
    drv = webdriver.Chrome('chromedriver')
    drv.get("http://naver.com")

    # 크롬창을 스크린샷
    drv.save_screenshot('001.png')
    drv.close()


# 서울시 구별 주유소 가격 정보 얻기
driver = webdriver.Chrome('chromedriver')
driver.get("http://www.opinet.co.kr/searRgSelect.do")
time.sleep(5)

# 지역선택상자의 id를 찾아 가져온다
# 시도 선택을 서울로 맞춘다
sido_list_raw = driver.find_element(By.XPATH, """//*[@id="SIDO_NM0"]""")
sido_list = sido_list_raw.find_elements(By.TAG_NAME, "option")

# 시도 리스트에 담긴 value 값을 추출
sido_names = [option.get_attribute('value') for option in sido_list]

# ''를 제거
sido_names.remove('')
print(sido_names)

element = driver.find_element(By.ID, "SIDO_NM0")
element.send_keys(sido_names[0])

# 구 선택상자의 xpath를 복사한다.
gu_list_raw = driver.find_element(By.XPATH, """//*[@id="SIGUNGU_NM0"]""")

# '서울'의'구'가 담겨 있는'option'태그를 모두 찾는다.
# 여기서 selenium은 복수형으로 elements를 써주어야 find_all과 같은 일을 한다.
# element라고 적으면 1개만 찾는다.
gu_list = gu_list_raw.find_elements(By.TAG_NAME, "option")

# 구 리스트에 담긴 value 값을 추출
gu_names = [option.get_attribute('value') for option in gu_list]

# ''를 제거
gu_names.remove('')
print(gu_names)

# 구 옵션에서 값을 1개 찾아서 0번째 값을 구 선택상자에 입력한다
element = driver.find_element(By.ID, "SIGUNGU_NM0")
element.send_keys(gu_names[0])

# 조회 버튼을 클릭하는 xpath
xpath = """//*[@id="searRgSelect"]"""
element_sel_gu = driver.find_element(By.XPATH, xpath)
element_sel_gu.click()

# 엑셀저장을 클릭하는 xpath
xpath = """//*[@id="glopopd_excel"]"""
element_get_excel = driver.find_element(By.XPATH, xpath)
element_get_excel.click()

# for 문을 돌려 모든 구에 대해 엑셀 데이터를 추출한다
for gu in tqdm_notebook(gu_names):
    # 구 옵션에서 값을 1개 찾아서 0번째 값을 구 선택상자에 입력한다
    element = driver.find_element(By.ID, "SIGUNGU_NM0")
    element.send_keys(gu)

    time.sleep(2)

    # 조회 버튼을 클릭하는 xpath
    xpath = """//*[@id="searRgSelect"]"""
    element_sel_gu = driver.find_element(By.XPATH, xpath)
    element_sel_gu.click()

    time.sleep(2)

    # 엑셀저장을 클릭하는 xpath
    xpath = """//*[@id="glopopd_excel"]"""
    element_get_excel = driver.find_element(By.XPATH, xpath)
    element_get_excel.click()

driver.close()
