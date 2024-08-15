n-gram 重叠比例用于衡量两个文本之间的相似性，尤其在自然语言处理中，它是一种常见的文本比较方法。n-gram 是指文本中连续的 n 个词或字符的组合，通过计算两个文本中的 n-gram 重叠比例，我们可以量化它们的相似程度。

## n-gram 重叠比例的理解
n-gram 是指将一个序列（如句子、段落）按照连续的 n 个元素分割的结果。对于词级别的 n-gram 来说，n 表示词的个数；对于字符级别的 n-gram，n 表示字符的个数。

例如，对于句子 "the quick brown fox"，我们可以计算 2-gram（也称为 bigram）：

- 1-gram（单个词）：["the", "quick", "brown", "fox"]
- 2-gram（连续两个词）：["the quick", "quick brown", "brown fox"]
- 3-gram（连续三个词）：["the quick brown", "quick brown fox"]
- n-gram 重叠比例表示两个文本之间 n-gram 重叠的比例，通常用于评估它们的相似性。其计算公式如下：

$$ n-gram overlap = \frac{num of overlap n-grams}{num of all unique n-grams} $$

这个比例值介于 0 和 1 之间，1 表示两个文本的 n-gram 完全相同，0 表示没有任何 n-gram 重叠。

## 计算步骤
- 生成 n-gram：对于两个文本，分别生成它们的 n-gram 集合。
- 计算重叠 n-gram：比较两个文本的 n-gram 集合，找到它们的重叠部分（即相同的 n-gram）。
- 计算 n-gram 重叠比例：通过计算重叠 n-gram 的数量与所有 n-gram 的总数量之间的比例来计算重叠比例。

## Python 示例代码
以下是基于词的 n-gram 重叠比例的 Python 示例代码：
```
from collections import Counter

def generate_ngrams(text, n):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

def ngram_overlap_ratio(text1, text2, n):
    # 生成 n-grams
    ngrams1 = generate_ngrams(text1, n)
    ngrams2 = generate_ngrams(text2, n)
    
    # 统计 n-grams 的频次
    counter1 = Counter(ngrams1)
    counter2 = Counter(ngrams2)
    
    # 计算 n-gram 的重叠
    overlap = sum((counter1 & counter2).values())  # 交集部分
    total = sum((counter1 | counter2).values())  # 并集部分
    
    # 计算重叠比例
    return overlap / total if total > 0 else 0.0

# 示例文本
text1 = "the quick brown fox jumps over the lazy dog"
text2 = "the quick brown dog jumps over the lazy fox"

# 计算 2-gram 重叠比例
n = 2
overlap_ratio = ngram_overlap_ratio(text1, text2, n)
print(f"2-gram 重叠比例: {overlap_ratio:.2f}")
```
## 代码说明
- generate_ngrams
  生成指定文本的 n-gram 列表。文本先通过 split() 方法转换为词列表，然后通过 zip 方法生成 n-gram。

- ngram_overlap_ratio
    - 首先生成两个文本的 n-gram 列表。
    - 然后使用 Counter 统计 n-gram 的频次。
    - 利用集合操作 & 和 | 分别计算 n-gram 的重叠部分（交集）和总数量（并集）。
    - 最后，计算重叠比例。
  
## 示例解释
假设有两个句子：

- text1 = "the quick brown fox jumps over the lazy dog"
- text2 = "the quick brown dog jumps over the lazy fox"

在计算 2-gram 的重叠时：

- text1 的 2-gram 为：["the quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over the", "the lazy", "lazy dog"]
- text2 的 2-gram 为：["the quick", "quick brown", "brown dog", "dog jumps", "jumps over", "over the", "the lazy", "lazy fox"]
- 两个句子共有 5 个 2-gram 重叠（"the quick", "quick brown", "jumps over", "over the", "the lazy"），并集总数为 10 个 n-gram。

因此，重叠比例为：

$$ overlap ratio = \frac{5}{10} = 0.5 $$

## n-gram 重叠比例的应用
n-gram 重叠比例常用于以下场景：

文本相似度计算：用于衡量两个文本的相似性，在机器翻译、文本摘要等领域广泛应用。
文档去重：通过计算 n-gram 重叠比例可以有效判断两个文档是否相似，帮助去除重复文档。
抄袭检测：通过计算文章片段的 n-gram 重叠比例，可以有效检测抄袭或引用情况。

## 总结
n-gram 重叠比例是一个简单而有效的度量文本相似度的方法。它通过比较两个文本中的 n-gram 数组的重叠情况，量化它们之间的相似性。在 NLP 任务中，它作为衡量相似度的重要工具，能够用于多个不同的实际应用场景。
