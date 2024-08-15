Unigram 分词 是另一种常用的子词分词算法，它与 BPE 和 WordPiece 的递归合并策略不同，基于确定一组最佳子词单元的概率模型。该方法由 Google 的研究人员提出，尤其用于像 SentencePiece 这样的子词分词工具中。

## Unigram 分词的基本原理
- 初始化词汇表：首先初始化一个包含所有可能的子词的词汇表，词汇表可以从初始数据中生成。这些子词可以是字符、n-gram 或者词。
- 模型优化：对于给定的训练数据和初始词汇表，Unigram 分词通过最大化子词序列的似然来找到最优的词汇表。
- 迭代过程：
    - 根据当前的词汇表，计算每个词被分解成不同子词的概率，找到最优的分解方式。
    - 计算每个子词的出现概率，删除低概率的子词并更新词汇表。
    - 继续迭代，直到模型收敛或者达到设定的词汇表大小。
- 贪婪分词：训练完模型后，在推理时，Unigram 分词使用贪婪算法，将给定的词分解为概率最高的子词序列。

## Unigram 分词的主要特点
- 全局优化：Unigram 分词通过最大化子词序列的似然，从全局角度优化词汇表，而不是像 BPE 或 WordPiece 那样通过局部合并。

- 删除策略：Unigram 分词会删除低概率的子词，保留高频和有用的子词，从而缩小词汇表的大小。

- 平衡子词和词元的选择：它能够在字符级别和词级别之间取得平衡，使得词汇表既能处理常见的词，也能处理低频或未知的词。

## Unigram 分词的 Python 实现
下面是一个简化的 Unigram 分词的 Python 实现示例。该实现包括了初始化词汇表、迭代优化、贪婪分词等步骤。
```
import random
from collections import defaultdict

class UnigramModel:
    def __init__(self, corpus, vocab_size):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.vocab = self.init_vocab()

    def init_vocab(self):
        # 初始化词汇表，包含所有可能的子词 (字符级)
        vocab = set()
        for word in self.corpus:
            for i in range(len(word)):
                for j in range(i + 1, len(word) + 1):
                    vocab.add(word[i:j])
        return list(vocab)

    def calculate_probabilities(self, word, vocab):
        # 计算每个词的最优分解方式及其概率
        n = len(word)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        best_split = [[] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(i):
                subword = word[j:i]
                if subword in vocab:
                    cost = dp[j] + 1  # 可以用概率进行替代
                    if cost < dp[i]:
                        dp[i] = cost
                        best_split[i] = best_split[j] + [subword]
        
        return best_split[-1] if dp[-1] != float('inf') else ["[UNK]"]

    def update_vocab(self):
        # 更新词汇表，通过删除低频子词的方式减少词汇表大小
        subword_count = defaultdict(int)
        for word in self.corpus:
            best_split = self.calculate_probabilities(word, self.vocab)
            for subword in best_split:
                subword_count[subword] += 1
        
        # 通过统计低频子词并删除
        sorted_vocab = sorted(self.vocab, key=lambda x: subword_count[x], reverse=True)
        self.vocab = sorted_vocab[:self.vocab_size]

    def train(self, num_iterations):
        # 迭代训练模型
        for iteration in range(num_iterations):
            self.update_vocab()
            print(f"Iteration {iteration + 1}: Vocabulary size = {len(self.vocab)}")

    def tokenize(self, word):
        # 使用训练好的模型进行分词
        return self.calculate_probabilities(word, self.vocab)


# 示例训练语料库
corpus = ["low", "lower", "newest", "widest", "new", "wide", "newer"]

# 训练 Unigram 分词模型
vocab_size = 20
model = UnigramModel(corpus, vocab_size)
model.train(num_iterations=10)

# 使用训练好的模型进行分词
word = "newest"
tokens = model.tokenize(word)
print("分词结果:", tokens)
```
## 代码解释
- init_vocab：从语料库中提取所有可能的子词，包括每个单词的字符和 n-gram 片段，作为初始词汇表。

- calculate_probabilities：计算每个单词的最优子词分解方式。该函数使用动态规划算法，尝试找到词汇表中匹配的最长子词。

- update_vocab：根据子词的使用频率更新词汇表，删除不常见的子词。此过程可以多次迭代，直到词汇表收敛到预定大小。

- train：主训练循环，每次迭代后根据子词的频率更新词汇表。

- tokenize：使用训练好的模型对单词进行分词，找到最优的子词组合。

## 运行结果示例
假设我们有以下语料库：
```
corpus = ["low", "lower", "newest", "widest", "new", "wide", "newer"]
```
在训练好 Unigram 模型之后，我们对 newest 这个词进行分词，输出结果可能为：
```
分词结果: ['new', 'est']
```

## Unigram 分词的应用
自然语言处理模型：Unigram 分词通常用于训练 NLP 模型，例如 BERT 和 GPT 等。它可以根据数据中出现的词频生成子词表，处理罕见词并提高模型的泛化能力。

SentencePiece 工具：Unigram 分词是 Google 的 SentencePiece 工具中的一种默认方法，适用于各种语言模型的预处理阶段。

文本生成：通过使用 Unigram 分词，生成模型可以处理更多的低频词并减少 OOV（词汇外）问题。

## 总结
Unigram 分词是一种基于概率的分词方法，通过最大化词汇表中子词序列的似然来优化词汇表。它在 NLP 模型中有着广泛的应用，特别是在处理低频词和未知词方面。与 BPE 和 WordPiece 相比，Unigram 更关注全局的子词概率分布，使用删除低概率子词的策略来优化词汇表。
