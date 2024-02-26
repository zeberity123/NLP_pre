# 1_4_attendant.py


members = '''김정훈/참석/숙박
김희자/참석/숙박
정운석/참석/숙박
손상연/참석/숙박
송은숙/참석/숙박
이상기/참석/숙박
안은하/참석/숙박
라경향/참석/숙박×
류제록/참석/숙박
최보금/참석/숙박
정연호/참석/숙박
김태임/참석/숙박
손종남/참석/숙박
강경삼/참석/숙박
이승준/참석/숙박
이준섭/참석/숙박
권민주/참석/숙박
임희재/참석/숙박
이충환/참석/숙박
김순태/참석/숙박
박용우/참석/숙박
박웅현/참석/숙박×
김영석/참석/숙박
이승창/참석/숙박
하이순/참석/숙박
박영경/참석/숙박
김숙희/참석/숙박
신동원/참석/숙박
최병국/참석/숙박x
최귀순/참석/숙박
김태은/참석/숙박
전은순/참석/숙박
임정연/참석/숙박
윤재문/참석/숙박
김완기/참석/숙박
김윤정/참석/숙박'''


w = str(members).split('\n')
name = []
attd = []
stay = []

for i in w:
    data = i.split('/')
    name.append(data[0])
    attd.append('O' if data[1] == '참석' else 'X')
    stay.append('O' if data[2] == '숙박' else 'X')

import pandas as pd
df = pd.DataFrame({'이름':name, '참석여부':attd, '숙박여부':stay})
print(df)

df.to_excel('test1.xlsx', index=False)