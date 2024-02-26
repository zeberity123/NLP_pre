# 1_1_library.py
import requests
import re

# 퀴즈
# 용인 흥덕도서관 노트북 열람실에서 빈 좌석을 알려주세요
url = 'http://211.251.214.176:8800/index.php?room_no=2'
response = requests.get(url)
# print(response)
# print(response.text)

# seats = re.findall(r'<font style="color:green;font-size:13pt;font-family:Arial"><b>([0-9]+)</b></font>', response.text)
seats = re.findall(r'<.+color:green.+><b>([0-9]+)</b></font>', response.text)
print(seats)
