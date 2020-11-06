import preprocess.pre as pre
import re
import pandas as pd
from pprint import pprint

#过滤代码片段
def codefilter(htmlstr):
    s = re.sub(r'(<code>)(\n|.)*?(</code>)', " ",htmlstr,re.S)
    return s

# df = pd.read_csv('analysislib/nlp/stanfordnlp.csv')
df = pd.read_csv('data/nlp.csv')

# temp = df[df['Id'] == 8752748]
# s = str(temp['Body'].values)
# print(s)
# s = codefilter(s)
# print(s)
# s = pre.processbody(s)
# print(s)

# s = re.sub('\n+', '', s)
# print(s)

for index, row in df.iterrows():
    title = str(row['Title'])
    tags = str(row['Tags'])
    body = row['Body']
    print(str(row['Id']))
    title = pre.processbody(title)
    print(title)
    print(codefilter(body))
    body = pre.processbody(body)
    # print(body)
    tags = pre.preprocesstag(tags)
