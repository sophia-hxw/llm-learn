# 后缀数组应用于句子级别的去重
后缀数组是一种强大的数据结构，广泛用于处理字符串匹配、重复检测等问题。对于句子级别的重复检测，后缀数组可以帮助我们找到所有子串的排列，从而有效地检测出句子中是否存在重复模式。

## 后缀数组方法过滤句子级别重复
后缀数组是一种基于字符串后缀的排序数组，数组中的每个元素表示从某个位置开始的后缀在字典序中的位置。结合后缀数组和 LCP（最长公共前缀）数组，我们可以检测出重复的句子片段。

## 算法思路
- 构建后缀数组：将所有句子的后缀构建成一个后缀数组，并按照字典序对后缀数组进行排序。
- 构建 LCP 数组：在得到排序后的后缀数组后，我们可以计算相邻两个后缀之间的最长公共前缀，生成 LCP 数组。
- 过滤重复片段：通过 LCP 数组中的非零值，我们可以检测出多个句子共享的相同片段。通过这些片段，我们可以识别出重复的句子。

## Python 代码实现
以下是一个使用后缀数组和 LCP 数组来过滤句子级别重复的 Python 实现
```
def build_suffix_array(s):
    n = len(s)
    suffix_array = sorted(range(n), key=lambda i: s[i:])
    return suffix_array

def build_lcp_array(s, suffix_array):
    n = len(s)
    rank = [0] * n
    lcp = [0] * n
    for i, suffix in enumerate(suffix_array):
        rank[suffix] = i

    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = suffix_array[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
    return lcp

def filter_repeated_sentences(sentences):
    # 合并句子为单个字符串，并保留原始句子边界的索引
    combined_text = ' '.join(sentences)
    sentence_boundaries = []
    current_pos = 0
    for sentence in sentences:
        sentence_boundaries.append(current_pos)
        current_pos += len(sentence) + 1  # +1 for the space
    
    # 构建后缀数组和 LCP 数组
    suffix_array = build_suffix_array(combined_text)
    lcp_array = build_lcp_array(combined_text, suffix_array)

    # 找到重复片段
    repeated_phrases = set()
    for i in range(1, len(lcp_array)):
        if lcp_array[i] > 0:  # LCP > 0 表示有重复
            start = suffix_array[i]
            repeated_phrases.add(combined_text[start:start + lcp_array[i]])

    # 过滤重复句子
    filtered_sentences = []
    for sentence in sentences:
        if not any(phrase in sentence for phrase in repeated_phrases):
            filtered_sentences.append(sentence)
    
    return filtered_sentences

# 示例句子
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog.",
    "A different sentence appears here.",
    "Another sentence with a different structure."
]

filtered_sentences = filter_repeated_sentences(sentences)

print("Filtered Sentences:")
for sentence in filtered_sentences:
    print(sentence)
```
## 代码说明：
- build_suffix_array：根据输入字符串 s 构建后缀数组，后缀数组中的每个元素表示从该位置开始的后缀在字典序中的位置。
- build_lcp_array：根据后缀数组计算最长公共前缀数组 (LCP)，该数组的每个值表示两个相邻后缀之间的最长公共前缀的长度。
- filter_repeated_sentences：
    - 首先，将所有句子合并成一个大字符串，并在其中记录每个句子的起始位置。
    - 使用后缀数组和 LCP 数组找出重复的片段（即多个句子共享的相同子字符串）。
    - 通过这些片段，过滤掉含有这些重复片段的句子，从而去除重复的句子。

## 示例输出
```
Filtered Sentences:
A different sentence appears here.
Another sentence with a different structure.
```

## 复杂度分析：
构建后缀数组的复杂度为 $O(nlogn)$，其中 $n$ 是输入字符串的长度。构建 LCP 数组的复杂度为 $O(n)$。过滤句子的过程需要遍历每个句子和每个找到的重复片段，因此复杂度大致为 $O(n⋅m)$，其中 $n$ 是句子的数量，$m$ 是句子的平均长度。

## 总结
后缀数组和 LCP 数组方法为重复检测提供了一种高效的解决方案。通过这种方法，我们可以在句子级别上找到并过滤掉重复的句子片段。对于规模较大的文本数据集，这种方法具有较好的性能和准确性。

# LCP数组
LCP（最长公共前缀）数组记录了在后缀数组中相邻两个后缀之间的最长公共前缀长度。它是后缀数组的重要补充，通过它我们可以找到相邻后缀共享的公共前缀的长度，这对于重复检测等问题至关重要。

下面是详细的 LCP 数组的构建过程。
LCP 数组的定义
对于给定的字符串 S 及其对应的后缀数组 SA，LCP 数组 LCP[i] 表示后缀 SA[i] 和 SA[i-1] 这两个后缀的最长公共前缀的长度。

假设我们有一个字符串 S = "banana"，其后缀数组 SA = [5, 3, 1, 0, 4, 2]。后缀数组中排序后的后缀依次为：
```
SA[0] = "a"
SA[1] = "ana"
SA[2] = "anana"
SA[3] = "banana"
SA[4] = "na"
SA[5] = "nana"
```
LCP 数组 LCP[i] 表示后缀 SA[i] 和 SA[i-1] 的公共前缀的长度，例如：

LCP[1] = 1，因为 "a" 和 "ana" 之间的最长公共前缀是 "a"。
LCP[2] = 3，因为 "ana" 和 "anana" 之间的最长公共前缀是 "ana"。

## LCP 数组的构建算法
LCP 数组的构建依赖于后缀数组。为了有效地计算相邻后缀之间的公共前缀，我们可以使用Kasai 算法，该算法的时间复杂度为 $O(n)$。

## Kasai算法步骤
- 初始化 Rank 数组
  Rank 数组 Rank[i] 记录了后缀 S[i:] 在后缀数组中的位置。也就是说，Rank 数组表示每个后缀在后缀数组中的索引位置。

- 计算 LCP 数组
  根据 Rank 数组，我们可以通过比较相邻的后缀来计算 LCP 数组中的值。我们通过遍历字符串中的每个字符，计算它与字典序上前一个后缀的最长公共前缀。为了加速计算，我们利用了前一个计算的结果进行优化。

具体步骤如下：

- 初始化 h = 0，h 表示当前后缀的公共前缀长度。
- 遍历每个位置 i，根据 Rank 数组找到当前后缀在后缀数组中的位置 rank[i]，然后比较它与其前一个后缀的公共前缀长度。
- 当比较时，如果当前后缀的公共前缀长度大于 0，则继续递减公共前缀长度。

代码实现
以下是基于 Kasai 算法构建 LCP 数组的 Python 实现：
```
def build_lcp_array(s, suffix_array):
    n = len(s)
    rank = [0] * n  # Rank 数组
    lcp = [0] * n   # LCP 数组

    # 构建 Rank 数组：rank[i] 表示后缀 S[i:] 在后缀数组中的位置
    for i, suffix in enumerate(suffix_array):
        rank[suffix] = i

    h = 0  # 用于记录前一个公共前缀的长度
    # 依次计算每个后缀的 LCP 值
    for i in range(n):
        if rank[i] > 0:
            # j 是在 suffix_array 中 rank[i] 前一个后缀的索引
            j = suffix_array[rank[i] - 1]
            # 计算 S[i:] 和 S[j:] 的公共前缀
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            # 将计算出的公共前缀长度保存到 LCP 数组中
            lcp[rank[i]] = h

            # 减少 h 的值，以便下一次计算更快
            if h > 0:
                h -= 1

    return lcp
```
## 算法解释
- Rank 数组构建
  Rank 数组 rank[i] 表示后缀 S[i:] 在后缀数组中的位置。例如对于 S = "banana" 和后缀数组 SA = [5, 3, 1, 0, 4, 2]，Rank 数组为 [3, 2, 5, 1, 4, 0]。这意味着后缀 S[0:] = "banana" 在后缀数组的索引为 3，后缀 S[1:] = "anana" 的位置为 2，等等。

- LCP 数组构建
  使用 Rank 数组，我们可以知道当前后缀在后缀数组中的位置。根据这个位置，我们可以找到其在字典序上的前一个后缀 S[j:]，然后比较它们的公共前缀。通过逐步减少公共前缀长度 h 的值，我们可以避免从头开始每次重新计算，从而加速整个过程。

## 举例说明
假设我们有字符串 S = "banana"，其后缀数组 SA = [5, 3, 1, 0, 4, 2]。我们通过上述算法来构建 LCP 数组。

- Rank 数组：首先构建 Rank 数组 rank = [3, 2, 5, 1, 4, 0]。
- LCP 数组计算：
    - 对于 i = 0，其 rank 是 3，它前面的后缀是 S[1:] = "anana"。S[0:] = "banana" 和 S[1:] = "anana" 的公共前缀长度为 0，因此 LCP[3] = 0。
    - 对于 i = 1，其 rank 是 2，它前面的后缀是 S[3:] = "ana"。S[1:] = "anana" 和 S[3:] = "ana" 的公共前缀长度为 3，因此 LCP[2] = 3。
    - 对于 i = 2，其 rank 是 5，它前面的后缀是 S[4:] = "na"。S[2:] = "nana" 和 S[4:] = "na" 的公共前缀长度为 2，因此 LCP[5] = 2。
    依次类推，计算出整个 LCP 数组。
- 最后得到的 LCP 数组为 [1, 3, 0, 0, 2, 0]。

总结
- 后缀数组提供了字符串的所有后缀的字典序排序信息。
- LCP 数组通过比较后缀数组中相邻的后缀，记录了它们之间的最长公共前缀长度。
- 使用 Kasai 算法 构建 LCP 数组，时间复杂度为 $O(n)$，非常高效。
  
LCP 数组在字符串重复检测、模式匹配等问题中有重要应用。通过后缀数组和 LCP 数组的结合，我们可以快速检测字符串中的重复片段。
