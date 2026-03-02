import json

# 定义文件路径
input_file = 'evaluation/DeepResearchGYM/queries/researchy_queries_sample_doc_click_100.jsonl'
output_file = 'evaluation/DeepResearchGYM/queries/researchy_queries_sample_doc_click_100_fix.jsonl'

# 读取原始数据文件
with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# 转换并写入新文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    for line in lines:
        # 解析每一行的 JSON 数据
        data = json.loads(line.strip())
        
        # 构建新的数据结构
        new_data = {
            "id": int(data["id"]),
            "topic": "unknown",  # 根据要求设定为 "unknown"
            "language": "en",    # 默认设定为中文
            "prompt": data["query"]
        }
        
        # 写入转换后的数据
        json.dump(new_data, outfile, ensure_ascii=False)
        outfile.write('\n')

print("数据转换完成，已保存为 " + output_file)