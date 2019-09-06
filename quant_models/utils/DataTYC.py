# -*- coding: utf-8 -*-

import os
import time
import math
import pandas
import urllib
import requests
import traceback
from bs4 import BeautifulSoup
from collections import defaultdict

headers = {}
cookies = {}
ckfiles = 0
tnfiles = 1


# 获取Cookie
# from Logger import Logger


def GetCookie():
    global ckfiles, tnfiles, headers, cookies
    file = "tyc%d.json" % ckfiles
    ckfiles = (ckfiles + 1) % tnfiles

    headers.clear()
    cookies.clear()
    with open("./data/cookie/%s" % file, "r") as file:
        while True:
            text_line = file.readline().strip('\n')
            if text_line:
                kv = text_line.split(":")
                headers[kv[0]] = kv[1]
            else:
                break
    mcookie = headers["Cookie"]
    for s in mcookie.split(";"):
        out = s.replace(" ", "").split("=")
        cookies[out[0]] = out[1]


# 获取URL
def GetInfo(webn):
    info = None
    try:
        url = 'http://www.tianyancha.com/search?key=%s' % urllib.parse.quote(webn)
        page = requests.get(url, headers=headers, cookies=cookies, timeout=5)
        soup = BeautifulSoup(page.content, 'lxml')
        tab = soup.select(".result-tips")[0]
        res = tab.select('.tips-num')[0]
        info = [webn, res.string]
        tab = soup.select(".result-list")[0]
        res = tab.select('.search-item')[0]
        info.append(tab.find("em").string)
        info.append(tab.find("a").get("href"))
        info.append(info[3].split("/")[-1])
        return info
    except:
        # print(traceback.print_exc())
        print("Fail:%s(%s)" % (webn, url))
        time.sleep(10)
    return None


# 获取专利等
def GetPatent(webn, sec, wdirs, ftype="patent"):
    # ftype
    # patent:专利信息
    # tminfo:商标信息
    # cpoyR:软件著作权
    # icp:网站备案
    # copyrightWorks:作品著作权
    try:
        ids = webn.split("/")[-1]
        dct = {"patent": ["patentCount", "patent", 7, 30],
               "tm": ["tmCount", "tmInfo", 8, 25],
               "icp": ["icpCount", "icp", 8, 200],
               "copyrs": ["cpoyRCount", "copyright", 8, 30],
               "copyrw": ["copyrightWorks", "copyrightWorks", 7, 10]}

        col = {"patent": ["序号", "申请公布日", "专利名称", "申请号", "申请公布号", "专利类型", "操作"],
               "tm": ["序号", "申请日期", "商标", "商标名称", "注册号", "国际分类", "商标状态", "操作"],
               "icp": ["序号", "审核时间", "网站名称", "网站首页", "域名", "备案号", "状态", "单位性质"],
               "copyrs": ["序号", "批准日期", "软件全称", "软件简称", "登记号", "分类号", "版本号", "操作"],
               "copyrw": ["序号", "作品名称", "登记号", "类别", "创作完成日期", "登记日期", "首次发布日期"]}

        numn = "nav-main-%s" % dct[ftype][0]
        flgn = "CompanyDetail.Daohang.knowledgeProperty.%s" % dct[ftype][0]
        tnum = dct[ftype][3]

        info = []
        page = requests.get(webn, headers=headers, cookies=cookies)
        soup = BeautifulSoup(page.content, 'lxml')
        flag = soup.find("div", attrs={"tyc-event-ch": flgn})
        if (flag.select("span")[0].string is None):
            print("%s(不存在数据):%s" % (sec, webn))
            out = pandas.DataFrame([], columns=col[ftype])
            out.to_excel(wdirs + "company/%s/%s.xlsx" % (ftype, sec))
            return out

        tab = soup.select("#" + numn)[0]
        res = tab.select('.data-title')[0]
        info.append(res.string)
        res = tab.select('.data-count')[0]
        info.append(int(res.string))
        print(info)

        pag = 0
        num = 0
        tmp = []
        out = []
        maxp = math.ceil(info[1] / tnum)
        if maxp > 50:
            maxp = maxp - 1
        while pag < maxp:
            pag = pag + 1
            url = "https://www.tianyancha.com/pagination/%s.xhtml?ps=%s&pn=%s&id=%s&_=" % (
                dct[ftype][1], str(tnum), str(pag), str(ids))
            print("%d/%d:%s" % (pag, maxp, url))
            time.sleep(0.1)
            page = requests.get(url, headers=headers, cookies=cookies)
            soup = BeautifulSoup(page.content, 'lxml')
            soup = soup.select("tbody")[0]
            res = soup.select("td")
            for s in res:
                tmp.append(s.string)
                num = num + 1
                if num % dct[ftype][2] == 0:
                    print(tmp)
                    out.append(tmp)
                    tmp = []

        out = pandas.DataFrame(out, columns=col[ftype])
        out.to_excel(wdirs + "company/%s/%s.xlsx" % (ftype, sec))
        return out
    except:
        print(traceback.print_exc())
        print("%s:%s" % (sec, webn))
        # GetCookie()
        return None


# 获取股权结构
def GetData(webn, url, seq, parent):
    if seq > 5:
        return

    # print("开始爬取：%s(%s)!" % (webn, parent))
    print("开始爬取：%s(%s)!" % (webn, parent))
    if url != '-':
        try:
            # 如果之前已经下载则继续后续工作
            fn = './output/%s.xlsx' % webn
            if os.path.exists(fn):
                # print("历史成功:%s(%s)!" % (webn, parent))
                print("历史成功:%s(%s)!" % (webn, parent))
                data = pandas.read_excel(fn)
                for ind, row in data.iterrows():
                    GetData(row['holder'], row['url'], seq + 1, webn)
                return

            page = requests.get(url, headers=headers)
            soup = BeautifulSoup(page.content, 'lxml')
            tab = soup.select("#nav-main-holderCount")[0].parent

            # col=['序号','股东','网址','认缴比例','认缴出资额','认缴时间']
            col = ['num', 'holder', 'url', 'ratio', 'share', 'time']
            data = []
            for tr in tab.find('tbody').find_all('tr'):
                num = 0
                row = []
                for td in tr.find_all('td'):
                    if num == 1:
                        pass
                    elif num == 2:
                        res = td.find('a')
                        row.append(res.get('title'))
                        row.append(res.get('href'))
                    else:
                        row.append(td.string)
                    num = num + 1
                data.append(row)

            data = pandas.DataFrame(data, columns=col)
            data["company"] = webn
            data["web"] = url
            data.to_excel(fn)
            print("当前成功:%s(%s)!" % (webn, parent))
            for ind, row in data.iterrows():
                GetData(row['holder'], row['url'], seq + 1, webn)
        except:
            # print(traceback.print_exc())
            # print("终极法人：%s(%s),网址：%s!" % (webn, parent, url))
            print("终极法人：%s(%s),网址：%s!" % (webn, parent, url))
    else:
        # print("终极法人:%s(%s)!" % (webn, parent))
        print("终极法人:%s(%s)!" % (webn, parent))
    return


def GetShare(webn):
    result = [defaultdict(float)] * 5

    i = 0
    for company in [webn]:
        share = defaultdict(float)
        fn = './output/%s.xlsx' % company
        if os.path.exists(fn):
            data = pandas.read_excel(fn)
            for ind, row in data.iterrows():
                try:
                    ratio = float(row['ratio'].replace("%", "")) / 100
                except:
                    ratio = 1.0
                share[row['holder']] += ratio
    result[i] = share

    for i in range(1, 5):
        share = defaultdict(float)
        share_pre = result[i - 1]
        for company in share_pre.keys():
            fn = './output/%s.xlsx' % company
            if os.path.exists(fn):
                data = pandas.read_excel(fn)
                for ind, row in data.iterrows():
                    try:
                        ratio = float(row['ratio'].replace("%", "")) / 100
                    except:
                        ratio = 1.0
                    share[row['holder']] += ratio * share_pre[company]
        result[i] = share

    final = result[0].copy()
    for i in range(1, 5):
        for key, val in result[i].items():
            final[key] += val

    for ss in range(5):
        for i in range(1, 5):
            share = defaultdict(float)
            for company in result[i - 1].keys():
                fn = './output/%s.xlsx' % company
                if os.path.exists(fn):
                    data = pandas.read_excel(fn)
                    for ind, row in data.iterrows():
                        try:
                            ratio = float(row['ratio'].replace("%", "")) / 100.0
                        except:
                            ratio = 1.0
                        share[row['holder']] += ratio * final[company]
            result[i] = share

        final = result[0].copy()
        for i in range(1, 5):
            for key, val in result[i].items():
                final[key] += val

    final = pandas.Series(final).sort_values(ascending=False)
    final.to_csv("./logger/share_%s.csv" % webn)


def GetListInfo(codes, slist, wdirs):
    with open(wdirs + 'StockPage_TYC.csv', 'r+', encoding="gbk")  as r:
        Get = pandas.read_csv(r, index_col=None)

    Get.columns = ["code", "name", "number", "find", "url", "ids"]
    with open(wdirs + "StockPage_TYC.csv", "a", encoding="gbk") as f:
        for i in range(len(codes)):
            if codes[i] in list(Get["code"].values):
                continue
            # print(i)
            print(i)
            info = GetInfo(slist[i])
            if info is not None:
                # print([codes[i]] + info)
                print([codes[i]] + info)
                f.write((','.join([codes[i]] + info) + "\n"))
                f.flush()
            time.sleep(0.5)

    with open(wdirs + 'StockPage_TYC.csv', 'r+', encoding="gbk")  as r:
        Out = pandas.read_csv(r, index_col=None)

    url = []
    for index, item in Out.iterrows():
        url.append(item["url"].split("/")[-1])

    Out["ids"] = url
    Out.to_csv(wdirs + "StockPage_TYC.csv", encoding='gbk', index=False)
    Out.to_excel(wdirs + "StockPage_TYC.xlsx")


def GetListPatent(data, wdirs, ftypes=["patent"]):
    for index, value in data.iterrows():
        for ftype in ftypes:
            if os.path.exists(wdirs + "company/%s/%s.xlsx" % (ftype, index)):
                continue
            else:
                # print(index)
                print(index)
                GetPatent(value["url"], index, wdirs, ftype)


if __name__ == '__main__':

    sess = requests.Session()
    print(sess.cookies.get_dict())
    res = sess.get("https://www.tianyancha.com/company/199557844")
    cookies = sess.cookies.get_dict()
    url = 'http://www.tianyancha.com/search?key=%s' % urllib.parse.quote("平安银行股份有限公司")
    data = {'username': 'kikiyu0814', 'password': '13602479715yu'}
    test_url = 'https://www.tianyancha.com/company/199557844'
    page = requests.get(test_url, cookies=cookies, timeout=5)
    soup = BeautifulSoup(page.content, 'lxml')
    print(soup)

