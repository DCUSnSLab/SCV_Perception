# 어떤 인자가 들어와도 에러 안 나게 다 받아주는(kwargs) 만능 함수
def print_log(content, *args, **kwargs):
    # 내용은 그냥 출력해줌
    print(content)