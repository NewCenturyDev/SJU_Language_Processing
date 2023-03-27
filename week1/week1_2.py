# 1주차-2: Konlpy로 한글 문장을 토크나이징 해 보기

from konlpy.tag import Okt
from time import perf_counter

# 전역 변수
# JVM_HOME = "/Library/Java/JavaVirtualMachines/jdk1.8.0_361.jdk/Contents/Home/bin/java"
JVM_HOME = "/Library/Java/JavaVirtualMachines/jdk1.8.0_361.jdk/Contents/Home/jre/lib/server/libjvm.dylib"
okt = Okt(jvmpath=JVM_HOME)


# 메인 함수
def main():
    print("시작")
    start_time = perf_counter()

    text = "한글 문장의 예시를 들어 보려고 합니다."
    # print(okt.morphs(text))
    print(okt.pos(text, join=True))

    finish_time = perf_counter()
    print("소요시간: " + str(finish_time - start_time) + "초")
    print("morphs 웰케 느려...")


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
