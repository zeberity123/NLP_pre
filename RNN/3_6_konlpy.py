# 3_6_konlpy.py
import konlpy
import time

print(konlpy.corpus.kolaw.fileids())
print(konlpy.corpus.kobill.fileids())

f = konlpy.corpus.kobill.open('1809890.txt')
print(f)
print(f.read())

kolaw = konlpy.corpus.kolaw.open('constitution.txt').read()
print(kolaw)

start = time.time()
tagger = konlpy.tag.Hannanum()
# tagger = konlpy.tag.Kkma()
pos = tagger.pos(kolaw)
print(pos[:10])
print('소요시간 :', time.time() - start)






