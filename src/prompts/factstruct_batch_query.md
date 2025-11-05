# FactStruct 批量查询生成 Prompt

你是一个研究助手。请为以下 {{node_count}} 个大纲节点分别生成一个精确的搜索查询。

## 节点列表
{% for node in nodes %}
{{ loop.index }}. 节点: '{{ node.title }}'{% if node.parent_context %}（上下文：{{ node.parent_context }}）{% endif %}
{% endfor %}

## 要求
1. 为每个节点生成一个精确、具体的搜索查询
2. 查询应该能够帮助检索到与该节点主题相关的文档
3. 如果节点有上下文信息，请在查询中体现
4. 输出格式必须严格按照以下格式：
查询 1: [查询内容]
查询 2: [查询内容]
...
查询 {{ node_count }}: [查询内容]

请只输出查询，每行一个，严格按照上述格式。

