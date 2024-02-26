# 1_2_imdb.py
import json
import re
import requests

# 퀴즈
# imdb top250 사이트로부터
# 영화 정보를 가져와서 예쁘게 출력하세요
f = open('data/imdb250.txt', 'r', encoding='utf-8')
imdb = f.read()
f.close()

# Accept:
# text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
# Accept-Encoding:
# gzip, deflate, br, zstd
# Accept-Language:
# ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7
# Connection:
# keep-alive
# Cookie:
# ad-id=A9s82mDzYE-0ukyACAgEypY; ad-privacy=0
# Host:
# aax-eu.amazon-adsystem.com
# Referer:
# https://www.imdb.com/
# Sec-Ch-Ua:
# "Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"
# Sec-Ch-Ua-Mobile:
# ?0
# Sec-Ch-Ua-Platform:
# "Windows"
# Sec-Fetch-Dest:
# iframe
# Sec-Fetch-Mode:
# navigate
# Sec-Fetch-Site:
# cross-site
# Upgrade-Insecure-Requests:
# 1
# User-Agent:
# Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36
# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'}
#
# url = 'https://www.imdb.com/chart/top/'
# response = requests.get(url, headers=headers)
# print(response)
# print(response.text)

# print(len(imdb))


item = ''' 
{"currentRank":1,
"node":{"id":"tt0111161",
        "titleText":{"text":"쇼생크 탈출","__typename":"TitleText"},
        "titleType":{"id":"movie",
                     "text":"Movie",
                     "canHaveEpisodes":false,
                     "displayableProperty":{"value":{"plainText":"","__typename":"Markdown"},
                                            "__typename":"DisplayableTitleTypeProperty"},
                     "__typename":"TitleType"},
        "originalTitleText":{"text":"The Shawshank Redemption","__typename":"TitleText"},
        "primaryImage":{"id":"rm2311309825",
                        "width":384,
                        "height":500,
                        "url":"https://m.media-amazon.com/images/M/MV5BOTM0NDcwYTctNTVkZi00YmVjLTk4ZDMtYjE4ODZmMTcyYmRmXkEyXkFqcGdeQXVyMTI5MzA0ODcy._V1_.jpg",
                        "caption":{"plainText":"Morgan Freeman and Tim Robbins in 쇼생크 탈출 (1994)","__typename":"Markdown"},
                        "__typename":"Image"},
        "releaseYear":{"year":1994,"endYear":null,"__typename":"YearRange"},
        "ratingsSummary":{"aggregateRating":9.3,"voteCount":2860477,"__typename":"RatingsSummary"},
        "runtime":{"seconds":8520,"__typename":"Runtime"},
        "certificate":{"rating":"15","__typename":"Certificate"},
        "canRate":{"isRatable":true,"__typename":"CanRate"},
        "titleGenres":{"genres":[{"genre":{"text":"Drama","__typename":"GenreItem"},
                                  "__typename":"TitleGenre"}],
                       "__typename":"TitleGenres"},
        "canHaveEpisodes":false,
        "plot":{"plotText":{"plainText":"Over the course of several years, two convicts form a friendship, seeking consolation and, eventually, redemption through basic compassion.",
                            "__typename":"Markdown"},
                "__typename":"Plot"},
        "latestTrailer":{"id":"vi3877612057","__typename":"Video"},
        "series":null,
        "__typename":"Title"},
"__typename":"ChartTitleEdge"}
'''

# import json
# movie = json.loads(item)
# print(movie.keys())
# print(movie['node'].keys())

movies = re.findall(r'{"currentRank":.+?"__typename":"ChartTitleEdge"}',
                    imdb, re.DOTALL)
# print(len(movies))
# print(movies[0])

# 퀴즈
# 1. json을 사용해서 예쁘게 출력하세요
# 2. 정규 표현식을 사용해서 예쁘게 출력하세요

for m in movies:
    item = json.loads(m)
    rank = item['currentRank']
    node = item['node']
    title = node['titleText']['text']
    year = node['releaseYear']['year']
    runtime = node['runtime']['seconds']
    ratings = node['ratingsSummary']['aggregateRating']
    # cert = node['certificate']
    # print(node.keys())
    # if rank == 52:
    #     print(type(cert))
    #     print(rank, cert)

    # print(rank, title, year, runtime, ratings)
    print('등수 :', rank)
    print('제목 :', title)
    print('연도 :', year)
    print('시간 :', '{}시간 {}분'.format(int(runtime) // 3600, int(runtime) % 3600 // 60))
    print('평점 :', ratings)
    print()


# for m in movies:
#     title = re.findall(r'"titleText":{"text":"(.+?)"', m)
#     print(title[0])


