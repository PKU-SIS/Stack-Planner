# planner.py

from src.tools.bocha_search.web_search_en import web_search
from baseline.Webweaver.utils import llm
from baseline.Webweaver.prompts import *
from baseline.Webweaver.memory_bank import MemoryBank
# prompts.py



class Planner:

    def __init__(self, memory_bank):

        self.memory = memory_bank
        self.outline = None

    def init_outline(self, question):

        prompt = INIT_OUTLINE_PROMPT.format(question=question)

        outline = llm(prompt)

        self.outline = outline

        return outline

    def generate_queries(self, question):

        prompt = QUERY_PROMPT.format(
            question=question,
            outline=self.outline
        )

        queries = llm(prompt)

        queries = queries.split("\n")

        clean_queries = []

        for q in queries:

            q = q.strip()

            if not q:
                continue

            # 去掉 "- "
            if q.startswith("-"):
                q = q[1:].strip()

            # 去掉 "1. "
            if q[0].isdigit():
                q = q.split(".", 1)[-1].strip()

            # 去掉引号
            q = q.strip('"')

            if q:
                clean_queries.append(q)

        return clean_queries

    # def search_and_store(self, queries):

    #     evidence_ids = []

    #     for q in queries:

    #         results = web_search(q, top_k=3)#后面改成3

    #         for r in results:

    #             content = r["content"]
    #             source = r["url"]

    #             summary = llm(
    #                 EVIDENCE_SUMMARY_PROMPT.format(
    #                     content=content
    #                 )
    #             )

    #             eid = self.memory.add(
    #                 summary=summary,
    #                 content=content,
    #                 source=source
    #             )

    #             evidence_ids.append(eid)

    #     return evidence_ids



    # 这个是修改后的 `search_and_store` 方法，返回的是 evidence_ids 和 research 字段
    def search_and_store(self, queries):
        evidence_ids = []

        # 清空 research 数据
        self.research = {}

        for q in queries:
            results = web_search(q, top_k=3)

            for r in results:
                content = r["content"]
                source = {
                    "title": r["title"],
                    "url": r["url"]
                }

                summary = llm(
                    EVIDENCE_SUMMARY_PROMPT.format(
                        content=content
                    )
                )

                # 获取 eid
                eid = self.memory.add(
                    summary=summary,
                    content=content,
                    source=source  # 确保 source 传递正确
                )

                evidence_ids.append(eid)

        return {
            "evidence_ids": evidence_ids,
            "research": self.memory.get_research()  # 返回包含 research 的数据
        }




    def refine_outline(self, question):

        evidence = self.memory.get_summary_block()

        prompt = REFINE_OUTLINE_PROMPT.format(
            question=question,
            outline=self.outline,
            evidence=evidence
        )

        self.outline = llm(prompt)

        return self.outline





def main():

    question = "What are the environmental impacts of offshore oil drilling?"

    memory = MemoryBank()

    planner = Planner(memory)

    print("\n===== STEP1: Init Outline =====\n")

    outline = planner.init_outline(question)
    print(outline)

    print("\n===== STEP2: Generate Queries =====\n")

    queries = planner.generate_queries(question)

    for q in queries:
        print("-", q)

    print("\n===== STEP3: Search =====\n")

    evidence_ids = planner.search_and_store(queries)
    # print("Evidence IDs(id):", evidence_ids["evidence_ids"])
    # exit()
    print("Evidence IDs:", evidence_ids)

    print("\n===== STEP4: Memory get_summary_block =====\n")

    # print(memory.dump())
    print(memory.get_summary_block())

    print("\n===== STEP5: Refine Outline =====\n")

    refined = planner.refine_outline(question)

    print(refined)


if __name__ == "__main__":
    main()