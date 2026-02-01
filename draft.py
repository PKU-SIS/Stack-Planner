from datetime import datetime
import re
import json
from src.factstruct.document import FactStructDocument

import re
import json
from datetime import datetime
from src.factstruct.document import FactStructDocument

# def wrap_raw_docs_to_factstruct(raw_docs, source_type="web"):
#     wrapped = []

#     if (
#         isinstance(raw_docs, list)
#         and len(raw_docs) == 1
#         and isinstance(raw_docs[0], str)
#         and "tool_name" in raw_docs[0]
#         and "web_search" in raw_docs[0]
#     ):
#         s = raw_docs[0]
#         try:
#             # 1️⃣ 提取 content='[...]' 部分
#             content_match = re.search(r"content='(.*)'", s)
#             if content_match:
#                 content_str = content_match.group(1)
#                 # 2️⃣ 替换转义双引号
#                 content_str = content_str.replace('\\"', '"')
#                 # 3️⃣ 转成 list
#                 content_list = json.loads(content_str)
                
#                 # 4️⃣ 找到 raw_results
#                 raw_docs = []
#                 for item in content_list:
#                     if isinstance(item, dict) and "raw_results" in item:
#                         raw_docs = item["raw_results"]
#                         break
#             else:
#                 raw_docs = []

#         except Exception as e:
#             print("文档转换失败:", e)
#             raw_docs = []

#     # 封装 FactStructDocument
#     for i, d in enumerate(raw_docs):
#         if not isinstance(d, dict):
#             continue

#         text = d.get("snippet", "")
#         text = re.sub(r"[。.]{2,}", "", text).strip()
#         if not text:
#             continue

#         doc = FactStructDocument(
#             id=f"doc_{hash(text)}_{i}",
#             cite_id=i + 1,
#             text=text,
#             source_type=source_type,
#             timestamp=datetime.now(),
#             url=d.get("link"),
#             title=d.get("title"),
#         )
#         wrapped.append(doc)

#     return wrapped

import re
import json
from datetime import datetime
from src.factstruct.document import FactStructDocument

def wrap_raw_docs_to_factstruct(raw_docs, source_type="web"):
    wrapped = []

    if (
        isinstance(raw_docs, list)
        and len(raw_docs) == 1
        and isinstance(raw_docs[0], str)
        and "tool_name" in raw_docs[0]
        and "web_search" in raw_docs[0]
    ):
        s = raw_docs[0]
        try:
            # 1️⃣ 提取 content='[...]' 部分（非贪婪匹配）
            content_match = re.search(r"content='(.*?)' name=", s, re.DOTALL)
            if content_match:
                content_str = content_match.group(1)
                # 2️⃣ 替换转义双引号
                content_str = content_str.replace('\\"', '"')
                # 3️⃣ 转成 list（异常捕获）
                try:
                    content_list = json.loads(content_str)
                except json.JSONDecodeError:
                    # 如果 JSONDecodeError，尝试解析到最后一个 JSON 对象
                    # 这里简单方法：用 eval 安全子集
                    import ast
                    content_list = ast.literal_eval(content_str)

                # 4️⃣ 找到 raw_results
                raw_docs = []
                for item in content_list:
                    if isinstance(item, dict) and "raw_results" in item:
                        raw_docs = item["raw_results"]
                        break
            else:
                raw_docs = []

        except Exception as e:
            # 改用单参数输出，避免 logger 报错
            print(f"文档转换失败: {e}")
            raw_docs = []

    # 封装 FactStructDocument
    for i, d in enumerate(raw_docs):
        if not isinstance(d, dict):
            continue

        text = d.get("snippet", "")
        text = re.sub(r"[。.]{2,}", "", text).strip()
        if not text:
            continue

        doc = FactStructDocument(
            id=f"doc_{hash(text)}_{i}",
            cite_id=i + 1,
            text=text,
            source_type=source_type,
            timestamp=datetime.now(),
            url=d.get("link"),
            title=d.get("title"),
        )
        wrapped.append(doc)

    return wrapped


def test_wrap_raw_docs_to_factstruct():
    docs = wrap_raw_docs_to_factstruct(
        raw_docs=fake_raw_docs,
        source_type="web"
    )

    print(f"生成文档数量: {len(docs)}\n")

    for d in docs:
        print("----")
        print("id:", d.id)
        print("cite_id:", d.cite_id)
        print("title:", d.title)
        print("url:", d.url)
        print("text:", d.text)


# fake_raw_docs = [
#     """{"tool_name": "web_search", "output": "content='[[{\\"url\\": \\"https://www.doc88.com/p-99659607980344.html\\", \\"content\\": \\"神经损伤与功能重....。\\"}, {\\"url\\": \\"https://www.doc88.com/p-9476308582393.html\\", \\"content\\": \\"下载积分。。。。【关键\\"}], {\\"raw_results\\": [{\\"link\\": \\"https://www.doc88.com/p-99659607980344.html\\", \\"title\\": \\"中性粒细胞在缺血性脑卒中的作用研究进展_江娜 - 道客巴巴\\", \\"snippet\\": \\"这是第一篇文章的摘要。。。\\"}, {\\"link\\": \\"https://www.doc88.com/p-9476308582393.html\\", \\"title\\": \\"中性粒细胞在缺血性脑卒中的作用及治疗前景 - 道客巴巴\\", \\"snippet\\": \\"这是第二篇文章的摘要。。。\\"}]}]'"}"""
# ]
fake_raw_docs = [
    """{
        "tool_name": "web_search",
        "output": "content='[
            [
                {\\"url\\": \\"https://www.doc88.com/p-99659607980344.html\\", \\"content\\": \\"神经负担。\\"},
                {\\"url\\": \\"https://sjssygncj.tjh.com.cn/CN/lexeme/showArticleByLexeme.do?articleID=1826\\", \\"content\\": \\"中性粒。\\"},
                {\\"url\\": \\"https://www.doc88.com/p-9476308582393.html\\", \\"content\\": \\"下载用。【关键\\"}
            ],
            {
                \\"raw_results\\": [
                    {\\"link\\": \\"https://www.doc88.com/p-99659607980344.html\\", \\"title\\": \\"中性粒细胞在缺血性脑卒中的作用研究进展_江娜 - 道客巴巴\\", \\"snippet\\": \\"神经负担。\\"},
                    {\\"link\\": \\"https://sjssygncj.tjh.com.cn/CN/lexeme/showArticleByLexeme.do?articleID=1826\\", \\"title\\": \\"中性粒细胞介导的炎症反应在急性缺血性脑卒中的研究进展\\", \\"snippet\\": \\"中性粒...\\"
                    },
                    {\\"link\\": \\"https://www.doc88.com/p-9476308582393.html\\", \\"title\\": \\"中性粒细胞在缺血性脑卒中的作用及治疗前景 - 道客巴巴\\", \\"snippet\\": \\"下载用。【关键\\"}
                ]
            }
        ]' name='web_search' tool_call_id='call_0230efd9c446480ea45118'"
    }"""
]



test_wrap_raw_docs_to_factstruct()
