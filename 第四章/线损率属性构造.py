import pandas as pd

inputfile = "/Users/wangduan/python_machine_learning/数据挖掘/data/electricity_data.xls"
outfile = "/Users/wangduan/python_machine_learning/数据挖掘/test/electricity_data.xls"
data = pd.read_excel(inputfile)
data[u'线损率'] = (data[u'供入电量']-data[u'供出电量'])/data[u'供入电量']
data.to_excel(outfile, index=False)