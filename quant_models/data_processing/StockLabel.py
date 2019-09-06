#Sample code from Tan to retrive the stock label from data center from wall streets
#
import os
import traceback

os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
import cx_Oracle
import pandas as pd


# connct=cx_Oracle.connect('fic_qry','fic_qry','172.253.33.65:1521/zxdb')

def GetLabelList():
    connct = cx_Oracle.connect('fic_qry', 'fic_qry', '172.253.33.65:1521/zxdb')
    try:
        olabel = pd.read_sql("select PLATE_ID,PLATE_NAME,PLATE_LBL from bd_admin.STKCN_WS_PLATE", connct)
        omap = {0: "主题(0)", 1: "行业(1)", 2: "风格(2)"}

        def fmap(k):
            return omap[k]

        olabel["PLATE_LBL"] = olabel["PLATE_LBL"].apply(fmap)
    except:
        pass
    connct.close()
    return olabel


def GetLabelShare(id, rqdata):
    connct = cx_Oracle.connect('fic_qry', 'fic_qry', '172.253.33.65:1521/zxdb')
    try:
        olabel = pd.read_sql(
            "select PLATE_ID,SECU_CODE,CORE_FLAG,SECU_DESC from bd_admin.STKCN_WS_PLATE_SECU where PLATE_ID=%d" % id,
            connct)

        def fmap(k):
            if k is None:
                return "N"
            if k > 0:
                return "Y"
            else:
                return "N"

        def fname(k):
            try:
                return rqdata.GetInstrumentSymbol(k)
            except:
                return k

        def fcode(k):
            return k.replace("SS", "SH")

        olabel["CORE_FLAG"] = olabel["CORE_FLAG"].apply(fmap)
        olabel["SECU_CODE"] = olabel["SECU_CODE"].apply(fcode)
        olabel["SECU_NAME"] = olabel["SECU_CODE"].apply(fname)
    except:
        pass
    connct.close()
    return olabel


def GetLabelNews(id):
    connct0 = cx_Oracle.connect('fic_qry', 'fic_qry', '172.253.33.65:1521/zxdb')
    try:
        def fstr(k):
            return k + ","

        def ftunc(k):
            if len(k) > 0:
                return k[0:-1]
            else:
                return k

        def ftime(k):
            return str(k)

        if id == 0:
            olabel = pd.read_sql(
                "select MSG_ID,PLATE_ID,TIT,URL,GRD_TIME,ABST from bd_admin.STKCN_WS_PLATE_MSG where GRD_TIME>to_date(to_char(sysdate-3,'yyyy/mm/dd'),'yyyy/mm/dd') order by GRD_TIME DESC",
                connct0)
        else:
            olabel = pd.read_sql(
                "select MSG_ID,PLATE_ID,TIT,URL,GRD_TIME,ABST from bd_admin.STKCN_WS_PLATE_MSG where PLATE_ID=%d and GRD_TIME>to_date(to_char(sysdate-3,'yyyy/mm/dd'),'yyyy/mm/dd') order by GRD_TIME DESC" % id,
                connct0)
        olabel.drop_duplicates("MSG_ID", inplace=True)
        olabel["GRD_TIME"] = olabel["GRD_TIME"].apply(ftime)

        if len(olabel) > 0:
            data = pd.read_sql("select PLATE_ID,PLATE_NAME from bd_admin.STKCN_WS_PLATE", connct0)
            data.set_index("PLATE_ID", inplace=True)
            olabel["PLATE_ID"] = data.reindex(list(olabel["PLATE_ID"]), fill_value="").values

        if len(olabel) > 0 and id != 0:
            lst = str(tuple(list(olabel["MSG_ID"]))).replace(",)", ")")
            ostock = pd.read_sql("select MSG_ID,SECU_SHT from bd_admin.STKCN_WS_MSG_SECU where MSG_ID in %s" % (lst),
                                 connct0)

            ostock.drop_duplicates(inplace=True)
            ostock["SECU_SHT"] = ostock["SECU_SHT"].apply(fstr)
            ostock = ostock.groupby(["MSG_ID"]).sum()
            ostock["SECU_SHT"] = ostock["SECU_SHT"].apply(ftunc)
            olabel["PLATE_ID"] = ostock.reindex(olabel["MSG_ID"].values, fill_value="").values

    except:
        print(traceback.print_exc())
        print("连接失败")

    connct0.close()
    return olabel


# print(GetLabelNews(16844702))

def GetStockNews(id):
    id_mod = id.replace(".SH", "")
    id_mod = id_mod.replace(".SZ", "")
    id_mod = id_mod.replace(".SS", "")
    connct0 = cx_Oracle.connect('fic_qry', 'fic_qry', '172.253.33.65:1521/zxdb')
    try:
        def fstr(k):
            return k + ","

        def ftunc(k):
            if len(k) > 0:
                return k[0:-1]
            else:
                return k

        def ftime(k):
            return str(k)

        olabel = pd.read_sql(
            "select A.MSG_ID,A.PLATE_ID,A.TIT,A.URL,A.GRD_TIME,A.ABST from bd_admin.STKCN_WS_PLATE_MSG A, bd_admin.STKCN_WS_MSG_SECU B where B.TRD_CODE='%s' and B.GRD_TIME>to_date(to_char(sysdate-365,'yyyy/mm/dd'),'yyyy/mm/dd') and A.MSG_ID=B.MSG_ID order by A.GRD_TIME DESC" % id_mod,
            connct0)
        olabel.drop_duplicates("MSG_ID", inplace=True)
        olabel["GRD_TIME"] = olabel["GRD_TIME"].apply(ftime)
        if len(olabel) > 0:
            data = pd.read_sql("select PLATE_ID,PLATE_NAME from bd_admin.STKCN_WS_PLATE", connct0)
            data.set_index("PLATE_ID", inplace=True)
            olabel["PLATE_ID"] = data.reindex(list(olabel["PLATE_ID"]), fill_value="").values
        else:
            olabel["PLATE_ID"] = []
    except:
        print(traceback.print_exc())
        print("连接失败")

    connct0.close()
    return olabel


def GetNewsInfo(id):
    connct0 = cx_Oracle.connect('fic_qry', 'fic_qry', '172.253.33.65:1521/zxdb')
    try:
        def fstr(k):
            return k + ","

        def ftunc(k):
            if len(k) > 0:
                return k[0:-1]
            else:
                return k

        def ftime(k):
            return str(k)

        olabel = pd.read_sql(
            "select MSG_ID,PLATE_ID,TIT,URL,ABST,GRD_TIME from bd_admin.STKCN_WS_PLATE_MSG where MSG_ID=%d" % id,
            connct0)
        olabel.drop_duplicates(["MSG_ID", "URL"], inplace=True)
        olabel["GRD_TIME"] = olabel["GRD_TIME"].apply(ftime)
        if len(olabel) > 0:
            lst = str(tuple(list(olabel["MSG_ID"]))).replace(",)", ")")
            ostock = pd.read_sql("select MSG_ID,SECU_SHT from bd_admin.STKCN_WS_MSG_SECU where MSG_ID in %s" % (lst),
                                 connct0)
            ostock.drop_duplicates(inplace=True)
            ostock["SECU_SHT"] = ostock["SECU_SHT"].apply(fstr)
            ostock = ostock.groupby(["MSG_ID"]).sum()
            ostock["SECU_SHT"] = ostock["SECU_SHT"].apply(ftunc)
            olabel["PLATE_ID"] = ostock.reindex(olabel["MSG_ID"].values, fill_value="").values
    except:
        print(traceback.print_exc())
        print("连接失败")

    connct0.close()
    return olabel


def GetNewsToday(id):
    connct0 = cx_Oracle.connect('fic_qry', 'fic_qry', '172.253.33.65:1521/zxdb')
    try:
        def fstr(k):
            return k + ","

        def ftunc(k):
            if len(k) > 0:
                return k[0:-1]
            else:
                return k

        def ftime(k):
            return str(k)

        olabel = pd.read_sql(
            "select HOT_SN,TIT,MSG_ID,FEA_MSG_TIT,FEA_MSG_ABST,PLATE_ID,FEA_MSG_CDT from bd_admin.STKCN_WS_TD_OPT where TODAY_DT=to_date(to_char(sysdate,'yyyy/mm/dd'),'yyyy/mm/dd')",
            connct0)
        olabel = olabel.sort_values(by="HOT_SN", ascending=True)
        olabel["FEA_MSG_CDT"] = olabel["FEA_MSG_CDT"].apply(ftime)
    except:
        print(traceback.print_exc())
        print("连接失败")

    connct0.close()
    return olabel


if __name__ == '__main__':
    print(GetLabelNews(16844702))
