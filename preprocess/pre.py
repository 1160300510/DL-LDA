import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer

#去除text中的所有非字母内容，包括标点符号、空格、换行、下划线等
def replace_all_blank(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    res = " ".join(text.split())  # 删除多余的空格
    return res

#判断是否为数字
def isNumber(word):
    matchobj = re.match(r"\d+", word, re.M|re.I)
    if matchobj:
        return True
    else:
        return False

#过滤代码片段
def codefilter(htmlstr):
    s = re.sub(r'(<code>)(\n|.)*?(</code>)', ' ', htmlstr, re.S)
    return s

#过滤html中的标签
def htmlfilter(htmlstr):
    # 先过滤CDATA
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)  # 匹配CDATA
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # Script
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # style
    re_br = re.compile('<br\s*?/?>')  # 处理换行
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    re_comment = re.compile('<!--[^>]*-->')  # HTML注释

    s = re_cdata.sub('', htmlstr)  # 去掉CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)  # 去掉style
    # s = re_br.sub('\n', s)  # 将br转换为换行
    s = re_br.sub(' ', s)
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_comment.sub('', s)  # 去掉HTML注释
    # 去掉多余的空行
    # blank_line = re.compile('\n+')
    s = re.sub(r'\\n(\B|\b)', ' ',s)
    return s
##替换常用HTML字符实体.
# 使用正常的字符替换HTML中特殊的字符实体.
# 你可以添加新的实体字符到CHAR_ENTITIES中,处理更多HTML字符实体.
# @param htmlstr HTML字符串.
def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', }

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        entity = sz.group()  # entity全称，如&gt;
        key = sz.group('name')  # 去除&;后entity,如&gt;为gt
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            # 以空串代替
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr

#去除停用词,小写,词干化
def preprocess(str):
    #小写
    str = str.lower()
    #去除数字，标点符号和非字母符号
    str = replace_all_blank(str)
    #分词
    words = word_tokenize(str)
    #词形还原和词干处理
    lemma = WordNetLemmatizer()
    stem = SnowballStemmer("english")
    wordlemmas = []
    for word in words:
        word = lemma.lemmatize(word)
        word = stem.stem(word)
        wordlemmas.append(word)
    #英语停用词和常用词
    english_stopwords = stopwords.words("english")
    commonworsd = ["word", "list", "using", "use", "like", "want", "get", "would", "follow", "tri", "word", "wa", "edu"
        , "com", "http", "way", "web"]
    words_stopwordsremoved = []
    for word in wordlemmas:
        if word not in english_stopwords and word not in commonworsd:
            words_stopwordsremoved.append(word)
    result = ''
    for word in words_stopwordsremoved:
        result += ' ' + word
    return result.strip()


#处理post的body部分
def processbody(str):
    str = replaceCharEntity(str)
    str = codefilter(str)
    str = htmlfilter(str)
    str = preprocess(str)
    return str

#处理post的tag部分
def preprocesstag(str):
    p1 = re.compile(r'[<](.*?)[>]', re.S)
    results = re.findall(p1, str)
    tag = ''
    for s in results:
        tag += s + ' '
    tag = tag[:-1]
    return tag


if __name__ == '__main__':
    df = pd.read_csv('../data/nlp.csv')
    for index, row in df.iterrows():
        title = str(row['Title'])
        tags = str(row['Tags'])
        body = str(row['Body'])
        title = processbody(title)
        print(title)
        body = processbody(body)
        tags = preprocesstag(tags)
        row['Title'] = title
        print(row['Title'])
        row['Tags'] = tags
        row['Body'] = body
        # if(len(body)<20 or len(body)>1000):
        #     df.drop(index=index, inplace=True, axis=0)
    df.to_csv('../preprocess/nlp-process.csv')
