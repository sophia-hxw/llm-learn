import re
from langdetect import detect
from spellchecker import SpellChecker

# 简单的拼写检查器
spell = SpellChecker()

def filter_data(data):
    filtered_data = []
    
    for sample in data:
        # 1. 语法/拼写检查：去除拼写错误超过一定阈值的句子
        words = sample.split()
        misspelled = spell.unknown(words)
        if len(misspelled) / len(words) > 0.3:  # 如果超过30%的单词拼写错误，则过滤
            continue
        
        # 2. 去除重复数据
        if sample in filtered_data:
            continue
        
        # 3. 过滤无信息样本 (长度过短或无意义的文本)
        if len(sample) < 10 or re.match(r"^\s*$", sample):  # 去除短句或纯空白
            continue
        
        # 4. 语言检测 (确保数据是特定语言，例如英语)
        try:
            language = detect(sample)
            if language != 'en':  # 假设我们只想保留英文数据
                continue
        except:
            continue
        
        # 如果数据通过了所有检查，则将其添加到过滤后的数据中
        filtered_data.append(sample)
    
    return filtered_data

# 示例数据
data = [
    "This is a correct sentence.",
    "Ths sntnce hs mny splling errs.",
    "数据乱码123",
    "啊啊啊",
    "Test sentence."
]

filtered_data = filter_data(data)
print(filtered_data)
