import os

# 파일 경로를 (폴더, 파일명, 확장자)로 나누는 단순 함수
def fileparts(path):
    if path is None:
        return None, None, None
    dirname = os.path.dirname(path)
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    return dirname, name, ext

# 혹시 몰라 자주 쓰이는 함수들도 빈 껍데기로 넣어둠
def load_txt_file(path): return []
def save_txt_file(data, path): pass
def mkdir_if_missing(path): pass