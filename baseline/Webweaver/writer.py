from baseline.Webweaver.utils import llm
from baseline.Webweaver.prompts import *
from baseline.Webweaver.memory_bank import MemoryBank
import re

class Writer:
    def __init__(self, memory_bank):
        self.memory = memory_bank
        self.used_citations = set() # 用于追踪报告中实际用到了哪些引用

    def write_section(self, section, evidence_ids):
        # 获取引用的内容
        evidence_text = self.memory.get_contents(evidence_ids)
        
        # 格式化引用，确保每一条证据文本前面都加上对应的 <citation> 标签
        formatted_evidence = []
        for i, evidence in enumerate(evidence_text):
            formatted_evidence.append(f"<citation>{evidence_ids[i]}</citation>{evidence}")

        # 创建包含证据的 prompt
        prompt = WRITE_SECTION_PROMPT.format(
            section=section,
            evidence="\n".join(formatted_evidence)  # 将格式化后的证据拼接到一起
        )
        
        # 打印调试信息
        # print("prompt:", prompt)
        # exit()  # 用于调试时暂停程序，查看 prompt 内容
        
        # 获取 LLM 生成的内容
        content = llm(prompt)
        
        # 记录本次使用的引用 ID
        for eid in evidence_ids:
            if f"<citation>{eid}</citation>" in content:
                self.used_citations.add(eid)
        
        return content

    def format_citations(self, text):
        """
        将文本中的 <citation>E0</citation> 替换为 [0]
        """
        # 使用正则匹配 <citation>E(\d+)</citation> 并替换为 [\1]
        return re.sub(r"<citation>E(\d+)</citation>", r"[\1]", text)

    def generate_references(self):
        """
        根据已使用的引用 ID 生成参考文献列表
        """
        if not self.used_citations:
            return ""

        ref_text = "\n\n## References\n"
        # 按编号排序输出
        sorted_eids = sorted(list(self.used_citations), key=lambda x: int(x[1:]))
        
        for eid in sorted_eids:
            # 获取 title 和 url
            # print("eid",eid)
            title = self.memory.research[eid]['title']  # 从 research 中获取 title
            url = self.memory.research[eid]['url']  # 从 research 中获取 url
            index = eid[1:]  # 提取 E0 中的 0
            ref_text += f"[{index}] {title}. {url}\n"  # 格式化参考文献条目
            
        return ref_text

    def write_report(self, sections):
        report = ""
        self.used_citations = set() # 重置引用记录
        print("sections",sections)
        for sec, eids in sections:
            # 1. 清理标题
            cleaned_section = self.clean_section_text(sec)
            
            # 2. 生成正文
            section_content = self.write_section(sec, eids)
            
            # 3. 转换引用格式 (例如 <citation>E0</citation> -> [0])
            formatted_content = self.format_citations(section_content)
            print("formatted_content",formatted_content)
            # 4. 拼接
            report += f"\n\n{cleaned_section}\n{formatted_content}"

        # 5. 最后加上参考文献部分
        report=self.clean_report_content(report)
        report=self.clean_using_llm(report)
        report += self.generate_references()
        return report

    def clean_section_text(self, text):
        # 匹配标题行：数字. **标题**
        match = re.search(r"^\d+\.\s?\*\*.*?\*\*", text, flags=re.MULTILINE)
        if match:
            return match.group(0).strip()
        return text.split('\n')[0].strip()


    def clean_report_content(self,content):
        # 移除多余的标签，如 <improved_outline> 和 </improved_outline>
        content = re.sub(r"</?improved_outline>", "", content)
        
        # 清理并合并多个引用
        content = re.sub(r"<citation>(.*?)</citation>", r"\1", content)  # 合并引用，移除标签
        content = re.sub(r"(\d+)\s*<citation>.*?</citation>", r"\1", content)  # 去除单个引用标签

        # 你可以根据需要扩展其他清理操作，比如移除不必要的换行或空格
        content = re.sub(r"\n+", "\n", content).strip()

        return content

    def clean_using_llm(self,text):
        # 向 LLM 提供清理命令，要求它处理多余的噪声、引用格式、标题问题等
        # clean_command = f"""
        # You are a professional reporter. Clean the following report by:
        # - Removing extra tags like <improved_outline> and ensuring no unnecessary markup.
        # - Fixing citation formatting so that all citations are correctly placed inline and merged where necessary. Citations should be in the format: "Source A【1】【2】", not multiple separate tags.
        # - Ensure the report has a clear, concise title that reflects the content accurately. You should generate a suitable title based on the report's theme and ensure it is placed at the beginning.
        # - The report should be free of unnecessary noise and look like a clean, professional report.

        # Report content: 
        # {text}
        # """
        clean_command=f"""
        You are a professional research editor. Your job is to CLEAN and FORMAT the report, NOT rewrite it.

        IMPORTANT PRINCIPLES:
        - DO NOT remove information.
        - DO NOT summarize or shorten the content.
        - DO NOT change the meaning of any paragraph.
        - DO NOT delete citations.
        - Preserve the original analysis depth and document length as much as possible.

        Your task is ONLY to improve formatting, citation style, and presentation quality.

        ================
        CLEANING TASKS
        ================

        1. Remove noise
        - Remove unnecessary tags such as <improved_outline>, <draft>, or similar markup.
        - Remove system artifacts or formatting noise.
        - Do NOT remove any meaningful text.

        2. Preserve full content
        - Keep ALL paragraphs and sections.
        - Do NOT merge sections aggressively.
        - Do NOT shorten explanations or analysis.

        3. Fix citation formatting
        All citations must follow this format:

        Source Name【1】【2】

        Rules:
        - Merge duplicate citation markers if necessary.
        - If multiple citations appear consecutively like 【1】【2】【3】 keep them together.
        - Do NOT delete citations.
        - Do NOT create new citations.
        - Do NOT change citation numbering.

        4. Title improvement
        If the document lacks a clear title:
        - Generate ONE concise and accurate title based on the report topic.

        If a title already exists:
        - Improve clarity but keep the same meaning.

        The title must:
        - Be placed at the very beginning
        - Be concise and descriptive
        - Reflect the report theme

        5. Improve readability without altering meaning
        You may:
        - Fix grammar
        - Fix spacing
        - Improve paragraph breaks
        - Improve heading formatting

        But you MUST NOT:
        - Add new analysis
        - Remove analysis
        - Change arguments
        - Reduce content length significantly

        6. Preserve structure
        Try to maintain the original section structure and ordering.

        7. Output format
        Return ONLY the cleaned report text.

        Do NOT include:
        - explanations
        - comments
        - metadata
        - markdown code blocks

        ================
        REPORT TO CLEAN
        ================

        {text}

        ================
        CLEANED REPORT
        ================
        """

        # 使用 LLM 清理文本
        cleaned_text = llm(clean_command)
        return cleaned_text

def parse_outline(outline):

    sections = []
    current_title = None
    current_block = []
    evidence_ids = set()

    for line in outline.split("\n"):

        line = line.rstrip()

        # 跳过空行
        if not line.strip():
            continue

        # 新的 section (例如 "1. **Introduction**")
        if re.match(r"^\d+\.", line):

            # 如果已有标题，保存当前 section
            if current_title:
                sections.append(
                    (
                        "\n".join(current_block),  # 保存当前 section 内容
                        list(evidence_ids)         # 保存该 section 相关的引用
                    )
                )

            # 新标题开始，初始化
            current_title = line
            current_block = [line]  # 将标题加入当前块
            evidence_ids = set()    # 清空当前证据引用集合
            continue

        # 将当前行加入当前 block
        current_block.append(line)

        # 查找所有 <citation> 标签中的证据 ID
        cites = re.findall(r"<citation>(E\d+)</citation>", line)

        # 添加所有找到的证据 ID
        for c in cites:
            evidence_ids.add(c)

    # 保存最后一个 section
    if current_block:
        sections.append(
            (
                "\n".join(current_block),
                list(evidence_ids)
            )
        )

    return sections


def main():

    memory = MemoryBank()

    # 使用字典传递 title 和 url
    id1 = memory.add(
        summary="Deepwater Horizon spill damage",
        content="The 2010 BP oil spill released about 4.9 million barrels of oil.",
        source={
            "title": "EPA Report on BP Oil Spill",
            "url": "https://www.epa.gov/deepwater-horizon"
        }
    )  # E0

    id2 = memory.add(
        summary="Environmental monitoring policies",
        content="Environmental impact assessments improve compliance monitoring.",
        source={
            "title": "EU Policy Report on Environmental Monitoring",
            "url": "https://eu-policy-report.com/environmental-monitoring"
        }
    )  # E1

    id3 = memory.add(
        summary="Low-impact drilling technology",
        content="Seawater drilling fluids reduce environmental damage.",
        source={
            "title": "Energy Journal on Low-impact Drilling",
            "url": "https://energyjournal.com/low-impact-drilling"
        }
    )  # E2

    id4 = memory.add(
        summary="Arctic drilling risks",
        content="Arctic drilling carries a high risk of large oil spills.",
        source={
            "title": "Arctic Council Report on Drilling Risks",
            "url": "https://arctic-council.com/drilling-risks"
        }
    )  # E3

    id5 = memory.add(
        summary="Methane monitoring",
        content="Aircraft sensors track methane emissions from drilling.",
        source={
            "title": "Atmospheric Research on Methane Monitoring",
            "url": "https://atmospheric-research.com/methane-monitoring"
        }
    )  # E4


    writer = Writer(memory)

#     outline = """1. **Introduction**  
#    - Overview of offshore oil drilling and its role in global energy production<citation>E0</citation>.  
#    - Importance of studying environmental impacts in the context of climate change and biodiversity conservation<citation>E1</citation>.  
#    - Purpose of the research: To analyze, synthesize, and evaluate the environmental consequences of offshore oil drilling.  

# 2. **Key Environmental Impacts**  
#    - **Oil Spills and Leaks**: Short- and long-term effects on marine ecosystems, wildlife, and coastal communities.  
#      - Case study: The 2010 BP oil spill released ~4.9 million barrels of oil, causing ~82,000 bird deaths, 6,000 sea turtles, and 25,900 marine mammals. Dispersants like Corexit exacerbated contamination with PAHs, affecting marine reproduction and survival. Oil persistence in sediments hindered vegetation and microbial recovery for decades <citation>E0</citation>.  
#    - **Habitat Destruction**: Disruption of benthic ecosystems, coral reefs, and migratory patterns of marine species.  
#      - Arctic drilling risks: A 75% chance of large spills in the Chukchi Sea could lead to mass mortality of polar bears, bowhead whales, and seabirds. Cleanup in icy Arctic waters is nearly impossible due to remote locations and harsh weather <citation>E3</citation>.  
#    - **Water and Air Pollution**: Release of toxic chemicals (e.g., heavy metals, dispersants) and greenhouse gases (e.g., methane, CO₂).  
#      - The BP spill released 4.9 million barrels of greenhouse gas emissions during extraction and combustion <citation>E0</citation>.  
#    - **Noise and Seismic Disturbances**: Impact of drilling operations and seismic surveys on marine mammals and fish behavior.  
#    - **Climate Change Contributions**: Lifecycle emissions from extraction to combustion.  

# 3. **Assessment Methods and Tools**  
#    - **Environmental Impact Assessments (EIAs)**: Regulatory frameworks and their effectiveness.  
#      - EIAs reduce project delays by 30% and compliance costs by 15% through real-time data analytics <citation>E1</citation>.  
#    - **Remote Sensing and Monitoring Technologies**: Use of satellites, drones, and underwater sensors to track pollution and ecological changes.  
#      - Aircraft-based surveys (e.g., FAAM 146) map methane and hydrocarbon distributions, enabling cost-effective compliance monitoring <citation>E4</citation>.  
#    - **Predictive Modeling**: Simulating oil spill trajectories, habitat degradation, and climate impacts.  
#      - Dispersion models estimate fugitive emissions and inform Arctic response plans <citation>E4</citation>.  
#    - **Case Study Analysis**: Historical incidents (e.g., Deepwater Horizon, BP oil spill) and their ecological aftermath.  
#      - The BP spill’s long-term impacts include reduced biodiversity and slow recovery of deep-sea corals <citation>E0</citation>.  

# 4. **Mitigation Strategies and Applications**  
#    - **Regulatory Policies**: International agreements (e.g., MARPOL, OSPAR) and national regulations.  
#      - Emerging trends include stricter carbon targets (80% of jurisdictions plan stricter rules by 2025) and AI monitoring adoption (45% in 2024) <citation>E1</citation>.  
#    - **Technological Innovations**: Advances in spill response, leak detection, and low-impact drilling techniques.  
#      - Seawater-based drilling fluids and subsea processing reduce freshwater use and carbon footprints. Case studies include Equinor’s subsea factory and BP’s low-salinity EOR <citation>E2</citation>.  
#    - **Restoration Efforts**: Habitat rehabilitation and community-based conservation programs.  
#      - The BP spill’s $65 billion cleanup and ongoing restoration efforts highlight the need for long-term ecological recovery <citation>E0</citation>.  
#    - **Transition to Renewable Energy**: Role of offshore oil drilling in the context of global decarbonization goals.  

# 5. **Challenges and Limitations**  
#    - **Technical and Logistical Barriers**: Deep-sea drilling complexities and cleanup limitations.  
#      - Arctic drilling faces insurmountable cleanup challenges due to ice and remoteness <citation>E3</citation>.  
#    - **Data Gaps**: Inconsistent monitoring, lack of long-term studies, and regional variability in impacts.  
#      - Predictive analytics lowers regulatory breaches by 40%, addressing data gaps <citation>E1</citation>.  
#    - **Economic and Political Factors**: Conflicts between energy demands, corporate interests, and environmental protection.  
#      - Non-compliance risks include fines averaging €50,000 per violation in Europe <citation>E1</citation>.  
#    - **Climate Trade-offs**: Balancing short-term energy needs with long-term climate stability.  

# 6. **Case Studies**  
#    - Analysis of specific regions (e.g., Gulf of Mexico, Arctic, Gulf of Guinea) to highlight regional differences in impacts and responses.  
#      - **BP Oil Spill (Gulf of Mexico)**: $10 billion in fisheries/tourism losses and persistent contamination <citation>E0</citation>.  
#      - **Arctic Drilling Risks**: Shell’s *Kulluk* incident (2012) revealed inadequate safety measures <citation>E3</citation>.  
#    - Lessons learned from past disasters and successful mitigation strategies.  

# 7. **Conclusion**  
#    - Summary of key findings on environmental risks and trade-offs.  
#    - Recommendations for policy, industry practices, and future research directions.  
#    - Implications for sustainable energy transitions and ecosystem preservation.  """
    outline = """1. **Introduction**  
   - Overview of offshore oil drilling and its role in global energy production<citation>E0</citation>.  
   - Importance of studying environmental impacts in the context of climate change and biodiversity conservation<citation>E1</citation>.  
   - Purpose of the research: To analyze, synthesize, and evaluate the environmental consequences of offshore oil drilling.  

2. **Key Environmental Impacts**  
   - **Oil Spills and Leaks**: Short- and long-term effects on marine ecosystems, wildlife, and coastal communities.  
     - Case study: The 2010 BP oil spill released ~4.9 million barrels of oil, causing ~82,000 bird deaths, 6,000 sea turtles, and 25,900 marine mammals. Dispersants like Corexit exacerbated contamination with PAHs, affecting marine reproduction and survival. Oil persistence in sediments hindered vegetation and microbial recovery for decades <citation>E0</citation>.  
   - **Habitat Destruction**: Disruption of benthic ecosystems, coral reefs, and migratory patterns of marine species.  
     - Arctic drilling risks: A 75% chance of large spills in the Chukchi Sea could lead to mass mortality of polar bears, bowhead whales, and seabirds. Cleanup in icy Arctic waters is nearly impossible due to remote locations and harsh weather <citation>E3</citation>.    

3. **Conclusion**  
   - Summary of key findings on environmental risks and trade-offs.  
   - Recommendations for policy, industry practices, and future research directions.  
   - Implications for sustainable energy transitions and ecosystem preservation.  """
    sections = parse_outline(outline)

    print("Parsed sections:")
    for s in sections:
        print(s)

    print("\n===== GENERATING REPORT =====")

    report = writer.write_report(sections)

    print(report)


if __name__ == "__main__":
    main()