

class MemoryBank:
    def __init__(self):
        self.data = {}
        self.counter = 0
        self.research = {}

    def add(self, summary, content, source):
        eid = f"E{self.counter}"
        self.counter += 1

        # 存储证据数据
        self.data[eid] = {
            "summary": summary,
            "content": content,
            "source": source
        }

        # 存储证据的 research 数据，结构化为 "1", "2" 这种形式
        # research_id = str(self.counter)
        self.research[eid] = {
            "type": "page",
            "title": source["title"],  # 假设 source 是一个 dict，包含 title 和 url
            "url": source["url"],
            "content": content,
        }

        return eid

    def get_summary_block(self):
        text = []
        for eid, item in self.data.items():
            block = f"""
            [{eid}]
            Source: {item['source']}
            Summary:
            {item['summary']}
            """
            text.append(block)

        return "\n".join(text)

    def get_contents(self, eids):
        texts = []
        for eid in eids:
            if eid in self.data:
                texts.append(self.data[eid]["content"])
        return texts

    def get_research(self):
        return self.research



class MemoryBank:
    def __init__(self):
        self.data = {}
        self.counter = 0
        self.research = {}

    def add(self, summary, content, source):
        eid = f"E{self.counter}"
        self.counter += 1

        # 存储证据数据
        self.data[eid] = {
            "summary": summary,
            "content": content,
            "source": source
        }

        # 存储证据的 research 数据，结构化为 "1", "2" 这种形式
        research_id = str(self.counter)#research_id]
        self.research[eid] = {
            "type": "page",
            "title": source["title"],  # 假设 source 是一个 dict，包含 title 和 url
            "url": source["url"],
            "content": content,
        }

        return eid

    def get_summary_block(self):
        text = []
        for eid, item in self.data.items():
            block = f"""
            [{eid}]
            Source: {item['source']}
            Summary:
            {item['summary']}
            """
            text.append(block)

        return "\n".join(text)

    def get_contents(self, eids):
        texts = []
        for eid in eids:
            if eid in self.data:
                texts.append(self.data[eid]["content"])
        return texts

    def get_research(self):
        return self.research


def main():
    # 创建 MemoryBank 实例
    memory = MemoryBank()

    print("=== Add Evidence ===")

    # 添加第一条证据
    id1 = memory.add(
        summary="LLMs are built on transformer architecture.",
        content="Large language models such as GPT use transformer-based neural networks for sequence modeling.",
        source={"title": "Transformer (machine learning model)", "url": "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)"}
    )

    # 添加第二条证据
    id2 = memory.add(
        summary="GPT models are pretrained on massive text corpora.",
        content="GPT models are trained on large-scale internet text to learn language patterns.",
        source={"title": "GPT Documentation", "url": "https://platform.openai.com/docs"}
    )

    print("Added:", id1, id2)

    print("\n=== Summary Block (for LLM prompt) ===")

    # 获取并打印所有证据的摘要块
    summary_block = memory.get_summary_block()
    print(summary_block)

    print("\n=== Retrieve Full Contents ===")

    # 获取并打印所有证据的内容
    contents = memory.get_contents([id1, id2])
    for c in contents:
        print("-", c)

    print("\n=== Research Information ===")

    # 获取并打印 research 信息
    research_data = memory.get_research()
    for key, value in research_data.items():
        print(f"Research {key}: {value}")

if __name__ == "__main__":
    main()

