'''
Created on 2019年5月19日

@author: jinglingzhiyu
'''
import requests
from bs4 import BeautifulSoup

def getHTMLText(url, timeout=50):
    try:
        r = requests.get(url, timeout=timeout)      #获取url网页对象
        r.raise_for_status()                        #判断是否产生异常
        r.encoding = r.apparent_encoding            #改变编码方式
        return r.text                               #返回网页内容(以字符串形式)
    except:
        return "Request Error"

