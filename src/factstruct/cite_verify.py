import re
from collections import defaultdict
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from .document import FactStructDocument
from datetime import datetime
from sentence_transformers import CrossEncoder




#蕴涵模型判断引用文章和生成文章的关系
def filter_content_by_relevant_docs(
    content: str,
    relevant_docs: list,
    semantic_cls,
):
    """
    判断 content 中每个带引用的句子
    是否被其对应的 relevant_docs 支持
    """

    # 1. 构建 cite_id -> FactStructDocument 映射
    citeid2doc = {
        doc.cite_id: doc
        for doc in relevant_docs
    }

    # 2. 初始化 NLI pipeline（只初始化一次）
    supported_statements = []

    # 3. 句子切分
    sentences = re.split(r'(?<=[。！?])', content)
    sentences = [s.strip() for s in sentences if s.strip()]


    for sentence in sentences:
        # 4. 提取引用编号，如 【39】
        refs = re.findall(r"[【\[](\d+)[】\]]", sentence)
        if not refs:
            continue

        refs = list(set(int(r) for r in refs))

        # 5. 去掉引用符号，作为 hypothesis
        hypothesis = re.sub(r"[【\[]\d+[】\]]", "", sentence).strip()

        for cite_id in refs:
            if cite_id not in citeid2doc:
                continue

            doc = citeid2doc[cite_id]
            premise = doc.text

            # 6. NLI 推理
            # result = semantic_cls(input=(premise, hypothesis))
            # scores = result["scores"]
            # labels = result["labels"]

            # max_idx = scores.index(max(scores))
            # label = labels[max_idx]
            # score = scores[max_idx]
            scores = semantic_cls.predict([(premise, hypothesis)])[0]

            label_mapping = ['contradiction', 'entailment', 'neutral']
            max_idx = scores.argmax()
            label = label_mapping[max_idx]
            score = float(scores[max_idx])


            # 7. 保存结果（字段对齐）
            supported_statements.append({
                "statement": sentence,
                "doc_id": doc.cite_id,   # 对齐 [39]
                "doc_uid": doc.id,
                "nli_label": label,
                "nli_score": score
            })

    return supported_statements


#根据蕴涵模型结果，对生成文章进行处理
def mark_content_with_support(
    content: str,
    nli_results: list
):
    """
    根据 NLI 结果，按 citation 级别标注不被支持的引用为【?】
    """

    # -------- Step 1: 构建 sentence -> cite_id -> is_supported --------
    sentence2cite_support = defaultdict(dict)

    for r in nli_results:
        sentence = r["statement"]
        cite_id = r["doc_id"]
        # is_supported = (r["nli_label"] == "蕴涵")
        is_supported = (r["nli_label"] == "entailment")

        # 同一个 citation 只要有一次蕴涵，就算支持
        prev = sentence2cite_support[sentence].get(cite_id, False)
        sentence2cite_support[sentence][cite_id] = prev or is_supported

    # -------- Step 2: 按句子重写 content --------
    sentences = re.split(r'(?<=[。！?])', content)
    sentences = [s for s in sentences if s.strip()]

    new_sentences = []

    for sent in sentences:
        # 没有引用 → 原样保留
        if not re.search(r"[【\[]\d+[】\]]", sent):
            new_sentences.append(sent)
            continue

        # 如果这个句子没有任何 NLI 结果 → 全标 ?
        if sent not in sentence2cite_support:
            marked = re.sub(r"[【\[]\d+[】\]]", "【?】", sent)
            new_sentences.append(marked)
            continue

        cite_support = sentence2cite_support[sent]

        # 按 citation 逐个替换
        def replace_cite(match):
            cite_id = int(match.group(1))
            if cite_support.get(cite_id, False):
                return match.group(0)   # 保留原引用，如【37】
            else:
                return "【?】"

        marked = re.sub(
            r"[【\[](\d+)[】\]]",
            replace_cite,
            sent
        )

        new_sentences.append(marked)

    return "".join(new_sentences)


def repair_unknown_citations(
    content: str,
    relevant_docs: list,
    semantic_cls,
    entail_threshold: float = 0.35
):
    """
    修复 content 中的【？】：
    - 只补充新的 citation，不重复已有的
    - 若无任何蕴涵 → 删除【？】
    """

    sentences = re.split(r'(?<=[。！？])', content)
    sentences = [s for s in sentences if s.strip()]

    new_sentences = []

    for sentence in sentences:
        # 没有【？】直接保留
        if not re.search(r"[【\[]\s*[？?]\s*[】\]]", sentence):
            new_sentences.append(sentence)
            continue

        # 1️⃣ 提取句子中【已经存在的 citation】
        existing_cite_ids = set(
            re.findall(r"[【\[](\d+)[】\]]", sentence)
        )

        # 2️⃣ 构造 hypothesis（去掉【？】）
        hypothesis = re.sub(
            r"[【\[]\s*[？?]\s*[】\]]",
            "",
            sentence
        ).strip()

        newly_supported = []

        # 3️⃣ repair：只补充不存在的 citation
        for doc in relevant_docs:
            cid = str(doc.cite_id)
            if cid in existing_cite_ids:
                continue  # ⭐ 核心：避免重复

            # result = semantic_cls(input=(doc.text, hypothesis))
            # entail_score = result["scores"][1]
            scores = semantic_cls.predict([(doc.text, hypothesis)])[0]
            entail_score = float(scores[1])  # index 1 = entailment

            if entail_score > entail_threshold:
                newly_supported.append(cid)

        if newly_supported:
            #只把“新补充的 citation”放到【？】位置
            cite_str = "".join(
                f"【{cid}】" for cid in sorted(set(newly_supported), key=int)
            )
            repaired = re.sub(
                r"[【\[]\s*[？?]\s*[】\]]",
                cite_str,
                sentence
            )
            new_sentences.append(repaired)
        else:
            # 没有任何新蕴涵 → 删除【？】
            cleaned = re.sub(
                r"[【\[]\s*[？?]\s*[】\]]",
                "",
                sentence
            )
            new_sentences.append(cleaned)
    return "".join(new_sentences)







def main():

    # ===== 示例输入（你之后可以替换成真实数据加载） =====
    content = """环烷酸锌是一种常见的抗蚀油脂添加剂，其化学结构属于金属皂类缓蚀剂。环烷酸锌通过与碳钢表面的金属原子发生配位作用，形成一层稳定的保护膜，从而抑制腐蚀反应的发生【36】。该化合物的分子中含有酯基（-COOR）和Zn²⁺离子，这些官能团在红外光谱中表现出强烈的C=O伸缩振动峰（约1740 cm⁻¹），表明其具有较高的红外活性【38】。然而，由于其分子结构中缺少芳香环或共轭双键体系，拉曼活性相对较低。 环烷酸锌与其他金属皂（如硬脂酸锌、油酸锌等）存在协同效应，能够进一步增强其缓蚀性能。研究表明，在防锈油配方中添加适量的环烷酸锌，并与其他极性添加剂复配使用时，可以显著提高碳钢表面的耐腐蚀能力【37】【39】。从光谱特性来看，这种复配体系中的红外活性主要来源于环烷酸锌本身的酯基（-COOR）以及其他添加剂中的极性基团（如-COO⁻、-NH₂等），表现出较强的红外响应；而由于这些成分均缺乏明显的共轭π电子体系或高度不对称的极性基团，使得整个复配体系的拉曼活性依然较弱。 此外，环烷酸锌在不同环境下的稳定性也受到广泛关注。实验表明，在中性和弱酸性环境下，环烷酸锌能够保持较好的热稳定性和化学稳定性，适合用于长期储存的防锈油产品中【39】。而在强酸性或高温条件下，其分解速率加快，可能导致缓蚀性能下降。因此，在实际应用中需根据具体工况条件选择合适的配方比例和使用方式。 综上所述，环烷酸锌作为抗蚀油脂添加剂之一，具有良好的红外活性，适用于通过红外光谱进行分子结构分析和吸附行为研究。然而，由于其分子结构对称性较高且缺乏芳香环或共轭体系，其拉曼活性较弱，限制了其在拉曼光谱中的应用价值。
        """
    relevant_docs = [
        FactStructDocument(
            id="doc_150416164661259118_38",
            cite_id=38,
            text="石油磺酸盐早在三十年代就被选作缓蚀剂使用,直至今天仍是首选的主要油溶性防锈添加剂之一。特别在国内几乎所有的防锈油脂都使用它,一般添加量在10%以下(见附表1几种国内主要的防锈油脂配方)。石油磺酸盐,可在石油加工白油时作副产品获得。即在从石油加工成白油过程中用发烟硫酸磺化除去白油中的芳香烃,芳香烃的磺化生成就是石油磺酸,然后用不同的氢氧化物中和,制成相应的盐类——石油磺酸盐。由于所选用的中和剂(例如氢氧化钡、氢氧化钠、氢氧化钙)不同,就能得到三种",
            source_type="page",
            timestamp=datetime(2025, 12, 13, 13, 15, 12, 426772),
            embedding=None,
            url="https://xuewen.cnki.net/CJFD-FSYF198003005.html",
            title="石油磺酸钡与防锈油-《腐蚀与防护》1980年03期-中国知网"
        ),

        FactStructDocument(
            id="doc_-456049905913296104_39",
            cite_id=39,
            text="目前使用较广、效果较好的有以下几类:磺酸盐(磺酸钙、磺酸钠和磺酸钡)、羧酸及其盐类(十二烯基丁二酸、环烷酸锌、N-油酰肌氨酸十八胺盐)、有机磷酸盐类、咪唑啉盐、酯型防锈剂(羊毛脂及羊毛脂皂、司苯-60或80、氧化石油脂)、杂环化合物(苯并三氮唑)、有机胺类等。水溶性防锈剂主要有:亚硝酸钠、重铬酸钾、磷酸三钠、磷酸氢二铵、苯甲酸钠、三乙醇胺等。一、防锈油之作用: 1.",
            source_type="page",
            timestamp=datetime(2025, 12, 13, 13, 15, 12, 426781),
            embedding=None,
            url="https://www.360docs.net/doc/b55862025.html",
            title="金属清洗剂 - 360文档中心"
        ),

        FactStructDocument(
            id="doc_1414329639007991770_37",
            cite_id=37,
            text="本品用做为防锈添加剂,乳化剂,有相当抗盐水浸渍能力和相当好的油溶性,它对黑色金属和黄铜防锈性能较好,可作为多种极性物质在油中的助溶剂。对手汗和水有较强的转换能力,和其它防锈添加剂复合使用,常用作工序间的清洗和防锈油、防锈脂、切削联系人:邓经理 电话:18638799810 手机:15515959896 总经理:杨总 邮箱:zzhailong@126.com 网址:www.",
            source_type="page",
            timestamp=datetime(2025, 12, 13, 13, 15, 12, 426753),
            embedding=None,
            url="http://www.zzhailong.cn/page105?product_id=191",
            title="石油磺酸钠"
        )
    ]


    # ===================================================
    #造蕴涵模型
    # nli_model_path="/data1/Yangzb/Model/nlp_structbert_nli_chinese-tiny"
    # semantic_cls = pipeline(
    #     Tasks.nli,
    #     nli_model_path,
    #     model_revision='master'
    # )
    semantic_cls = CrossEncoder(
        "/data1/Yangzb/Model/nli-deberta-v3-small"
    )

    #这个是判断引用和句子的关系
    supported = filter_content_by_relevant_docs(
        content=content,
        relevant_docs=relevant_docs,
        semantic_cls=semantic_cls
    )
    print(supported)
    new_content = mark_content_with_support(
        content=content,
        nli_results=supported
    )

    print(new_content)

    repair_content=repair_unknown_citations(
        content=new_content,
        relevant_docs=relevant_docs,
        semantic_cls=semantic_cls
    )
    
    print(repair_content)


if __name__ == "__main__":
    main()
