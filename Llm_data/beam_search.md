Beam Search 是一种启发式搜索算法，通常用于自然语言处理任务中的序列生成问题，例如机器翻译、图像描述生成等。它是一种改进的宽度优先搜索算法，相比贪心搜索能有效地探索多个可能的输出路径，但又比完全的暴力搜索更加高效。

## Beam Search的工作原理
Beam Search 的核心思想是保留一组得分最高的部分解（称为beam），在每一步扩展这些部分解，直到生成完整的解。在每一步，它只保留一组得分最好的候选解，而不是所有的可能解，从而限制了搜索空间，避免了指数级的复杂度增长。

步骤：
- 初始化：在序列生成的初始时刻，Beam Search 会从空序列开始。
- 扩展候选：在每一步，算法会对当前保留的候选序列进行扩展，生成新的候选序列。通常，序列的扩展是通过预测模型给出的每个可能下一个元素来完成。
- 保留 Top-k：在扩展后，Beam Search 会计算每个候选序列的得分（通常是概率对数的加和），然后只保留得分最高的 k 个候选序列。这个 k 就是 beam size，决定了在每一步中保留的候选序列的数量。
- 重复扩展和剪枝：重复步骤 2 和步骤 3，直到生成完整的序列（通常是遇到结束符或达到最大长度）。
- 选择最佳序列：最终从所有的候选序列中选择得分最高的作为最终输出。

## Beam Size
Beam size 决定了每一步保留多少个候选序列。如果 beam size 越大，Beam Search 会保留更多的候选路径，从而可能生成更好的最终解，但也会增加计算成本和内存使用。相反，较小的 beam size 会使搜索速度更快，但可能错过较好的解。

- beam size = 1 时，Beam Search 退化为贪心搜索，只保留当前得分最高的一个候选序列。
- beam size 越大，算法会更接近于暴力搜索，因为它会考虑更多的可能序列。

## 伪代码示例
以下是 Beam Search 的简单伪代码示例：
```
function BEAM_SEARCH(beam_size):
    Initialize an empty priority queue (beam)
    Add the initial state (empty sequence) to the beam
    
    while beam is not empty:
        Create an empty list of candidates
        
        for each sequence in beam:
            for each possible next token:
                Generate a new candidate sequence by appending the token
                Compute the score of the new sequence (log-probabilities sum)
                Add the new candidate sequence to candidates
        
        Sort candidates by score in descending order
        Select the top beam_size candidates and set them as the new beam
        
        If the top sequence is complete (reaches end of sentence or max length):
            Return the best sequence
        
    Return the sequence with the highest score
```

## 应用场景
Beam Search 通常用于以下场景：

- 机器翻译：在生成译文时，Beam Search 可以探索多种可能的翻译路径，选择得分最高的翻译。
- 自动摘要：在自动生成文本摘要时，Beam Search 可以有效生成较为流畅且合理的摘要。
- 语音识别：Beam Search 可以帮助识别概率最高的词序列，从而提高识别的准确度。
- 图像描述生成：在描述图像内容时，Beam Search 可以为图像生成最优的文本描述。

## 示例：使用 Beam Search 生成句子
假设我们有一个语言模型，它可以给定一个上下文预测下一个单词的概率，使用 Beam Search 来生成句子：
```
import numpy as np

def beam_search_decoder(probabilities, beam_size):
    sequences = [[list(), 1.0]]  # 初始化 (空序列, 概率为1.0)
    
    for row in probabilities:  # 遍历每个时间步的概率分布
        all_candidates = list()
        
        for seq, score in sequences:
            for j, prob in enumerate(row):
                candidate = [seq + [j], score * -np.log(prob)]  # 累乘负对数概率作为得分
                all_candidates.append(candidate)
        
        # 按得分排序并保留 top k (beam size) 的候选
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_size]
    
    return sequences

# 示例：每个时间步的词汇概率分布 (假设4个单词的概率)
probabilities = [
    [0.1, 0.5, 0.2, 0.2],  # 第一个词的概率分布
    [0.2, 0.1, 0.6, 0.1],  # 第二个词的概率分布
    [0.25, 0.25, 0.25, 0.25]  # 第三个词的概率分布
]

beam_size = 2  # beam size 为 2
sequences = beam_search_decoder(probabilities, beam_size)

for seq, score in sequences:
    print(f"Sequence: {seq}, Score: {score}")
```

## 优势与劣势
- 优势：
比贪心搜索更全局化，能够探索多个可能的候选序列，找到得分最高的最终序列。
能有效地平衡搜索空间大小与计算效率。

- 劣势：
并不能保证找到最优解（因为 Beam Search 并不是穷尽所有可能性）。
Beam size 增加时，计算复杂度增加显著，需要更多的内存和时间。

## 总结
Beam Search 是一种用于序列生成的启发式搜索算法，通过保留多个候选解，可以生成比贪心搜索更好的结果。它在自然语言处理、语音识别等任务中被广泛应用，并且通过调整 beam size 可以在质量和效率之间进行权衡。