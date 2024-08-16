justext 是一种用于从网页中提取正文文本的方法，特别适合处理包含很多广告、导航栏、页脚等噪音内容的网页。它的主要原理是通过分析 HTML 文档中每段文本的长度和标签，判断这些文本块是否属于正文内容。

## Justext 工作原理
- 段落分类：根据 HTML 文档的结构，justext 将每一个文本块（通常是 HTML 段落 <p> 标签）单独提取出来，并基于文本长度、字数密度、HTML 标签类型、链接密度等特征，进行分类。

- 噪声过滤：算法会判断哪些段落属于噪声（如导航条、广告、版权声明等），并将其过滤掉。其核心是通过设定阈值来识别并排除具有高链接密度、短文本长度或常见于非正文部分的标签。

- 语言识别：算法通过语言模型判断网页的语言，以此来更精确地判断哪些内容属于正文。

- 输出纯文本：提取后的正文内容会输出为纯文本，方便后续处理。

## Python 实现
justext 可以通过 Python 中的第三方库实现。你可以通过如下步骤使用 justext 提取网页的正文内容。

1. 安装 justext
可以通过 pip 安装：
```
pip install justext
```
2. 使用 justext 提取网页正文
以下是一个简单的代码示例，它从网页中提取纯文本：
```
import requests
import justext

# 请求网页内容
response = requests.get('https://example.com')

# 使用 justext 提取正文
paragraphs = justext.justext(response.content, justext.get_stoplist("English"))

# 输出过滤后的正文文本
for paragraph in paragraphs:
    if not paragraph.is_boilerplate:
        print(paragraph.text)
```
## 解释：
- requests.get(): 发送 HTTP 请求获取网页的 HTML 内容。
- justext.justext(): 分析网页内容，并将网页中的段落分类。
- is_boilerplate: 用于识别非正文的段落（例如导航条、广告等），过滤掉这些段落后即可获取纯正文文本。

## 适用场景
justext 非常适合需要从网页中提取干净、无广告的正文内容的场景，常用于网络抓取、信息抽取、自然语言处理等任务。

你可以根据实际需求调整使用的语言模型（stoplist）以及其他参数，优化提取结果。

