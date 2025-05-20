# SYSTEM PROMPT

你现在是一名资深的软件工程师，你熟悉多种编程语言和开发框架，特别是Python。

你擅长解决技术问题，并具有优秀的逻辑思维能力。请在这个角色下为我解答以下问题。

# USER PROMPT

我需要你实现下面的一个程序：

1. 将每一个文章进行总结（文章级别），以及逐section进行总结，以构建不同层次的文章评述。
2. 对每个文章尝试找到它的关键词（Tag），这个关键词包含了其解决的问题、idea、用到的技术等等。
3. 每一个总结都需要向量化，存储到vector database中（或者embedding），方便后续进行查询，构建简单的“数据库”。
4. 对每个Tag，生成相应的总结：包含技术思想和实现、以及相关的文章。

在你进行总结、评论的时候，可以使用到先前的评论和总结进行迭代更新。我可能多次运行 main 函数，你需要复用先前的content。因此需要进行持久存储。（下面的目录结构）

我需要一个基础、minimal的实现。并不希望用非常复杂的database等技术。你只需要写出每个函数做了什么，并不需要给出实现。

### Code Instructions

1. 你可以使用 `"""..."""` 来赋值那些冗长的prompt。
2. 你可以基于OpenAI python sdk和numpy框架来实现，并且需要进行合理的多文件分割，进行模块化设计。
3. 你不需要写出非常标准的面向对象代码，我需要一个基础的实现即可。
4. 你可以对每一个函数都写一个注释说明它的作用。例如`"""Loads a numpy array (embedding) from an .npz file."""`，但你不需要精确到每一个参数的含义。

#### Embedding and Retrieval Augmented Generation

你可以通过embedding查询相关的信息，然后将其添加到当前的prompt中。

1. 第一次运行时，可能没有相关的信息。
2. 但是在后续运行时，你需要利用上这些信息。

### 目录结构

vault/
- docs/
    - paper1/
        - raw.pdf             # You can assume this file exists.
        - extracted.md   # This file has a same content compared with raw.pdf. You can assume this file exists.
        - summarized.md
        - sections_summarized.md # 逐个section总结，文件内使用 `<!-- SEPERATOR -->` 来标记不同的section
        - tldr.md
    - ...
- tags/
    - tag1/
        - description.md # wikipedia-like description. might talk about the related works, but short and clear.
        - survey.md # Talk about the related papers in the `docs/`.

其中，`paper1` 和 `tag1` 需要按合适的方法生成这个名字。例如文章的title和tag的name。

#### vector database 结构

需要对包含所有的markdown文档计算embedding。

你需要将其存储成 `npy` 格式的向量，按同样的名字放在与文档相同的文件夹中。例如：

vault/
- docs/
    - paper1/
        - raw.pdf             # You can assume this file exists.
        - extracted.md   # This file has a same content compared with raw.pdf. You can assume this file exists.
        - extracted.npz
        - summarized.md
        - summarized.npz

### 关键数据结构

你**必须**使用基于TypedDict的类，而非直接的json作为数据结构。

```python
from typing import TypedDict, List, Dict, Optional
import numpy as np

# 核心数据结构
class Content(TypedDict):
    content: str
    vector: np.ndarray  # 使用numpy的ndarray存储向量
class ArticleSummary(TypedDict):
    title: str
    content: Content  # 原始内容（extracted.md）
    summary: Content  # 文章级别总结
    tldr: Content  # 1～2句话总结。（约100～200词）
    section_summaries: List[Content]  # 按section的总结列表
    tags: List[str]  # 提取的关键词（Tag）
class TagInfo(TypedDict):
    name: str
    description: Content  # 标签描述（类似维基百科）
    survey: Content  # 相关论文综述
```

### Instructions

首先，你需要理解我的需求，并且给出你的每个文件需要实现的内容、数据结构的基本定义。（使用TypedDict进行实现，我已经给出一个基本的版本）（Identify the information that must be represented and how it is represented in the chosen programming language. Formulate data definitions and illustrate them with examples.）

其次，你需要分析需要哪些函数来实现上面的功能。函数要较为简洁，可以复用。（State what kind of data the desired function consumes and produces. Formulate a concise answer to the question what the function computes. Define a stub that lives up to the signature.）

然后，你可以适当的给出一些非常小的例子来辅助你后续的实现，这一步不是必须的，你可以跳过它（或者部分）。（Work through examples that illustrate the function’s purpose. Translate the data definitions into an outline of the function.)

最后，你给出最终我需要的结果，即逐个文件给出**完整**的实现，每个函数都必须已经实现。（Fill in the gaps in the function template. Exploit the purpose statement and the examples. **Implement the details**.）
