from transformers import AutoTokenizer
from collections import defaultdict

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens."
]

# 使用gpt2 tokenizer将输入分解为单词
tokenizer = AutoTokenizer.from_pretrained("gpt2")

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, _ in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

# 计算基础词典， 这里使用语料库中的所有字符
alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

# 增加特殊 Token 在字典的开头 ， gpt2中只有一个， Token``</endoftext/>``表示文本结束
vocab = ["</endoftext/>"] + alphabet.copy()

# 将单词切分为字符
splits = {word: [c for c in word] for word in word_freqs.keys()}

# 计算字典中所有词元对的频率
def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split)-1):
            pair = (split[i], split[i+1])
            pair_freqs[pair] += freq
    return pair_freqs

# 合并词元对
def merge_pair(a, b, splits, word_freqs):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        
        i = 0
        while i < len(split) -1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

# 迭代训练，每次选取的粉最高词元对进行合并，直到词典大小达到设置目标为止
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits, word_freqs)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits, word_freqs)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

# 训练完成后，tokenize函数用于给定文本进行词元切分
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, _ in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i+1] == pair[1]:
                    split = split[:i] + [merge] + split[i+2:]
                else:
                    i += 1
            splits[idx] = split
    return sum(splits, [])

tokenize("This is not a token.")




