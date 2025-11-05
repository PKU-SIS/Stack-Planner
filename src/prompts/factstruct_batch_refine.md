# FactStruct 批量修纲 Prompt

你是一个研究助手。我们刚刚检索了 {{ task_count }} 个节点，获得了新信息。你的任务是根据这些新信息，对大纲进行 {{ task_count }} 次 *独立的局部优化*。

## 当前研究大纲
{{ current_outline }}

{% for task in optimization_tasks %}
{{ task }}
{% endfor %}

## 要求
1. 对每个优化任务，**独立地**进行局部修改
2. 修改时只影响目标节点及其子节点，不要影响其他不相关的节点
3. 修改后的大纲应该保持层次结构清晰
4. 输出格式必须是 JSON，结构如下：
{
    "title": "根节点标题",
    "children": [
        {
            "title": "子节点1标题",
            "children": []
        },
        {
            "title": "子节点2标题（可能已修改）",
            "children": [
                {
                    "title": "新增或修改的子节点",
                    "children": []
                }
            ]
        }
    ]
}

请只输出 JSON，不要包含其他解释性文字。输出完整的修订后大纲树。

