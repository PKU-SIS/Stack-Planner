import requests
import os
import socket
import json
import time
from tqdm import tqdm

url_base = "https://ragflow.pkubir.cn/v1/document_api/"

kb_id = "39ea834abf1111f0bf2ecd6543f8a381"
tenant_id = "81ca9bf7bf1011f0bf2ecd6543f8a381"


def fetch_all_docs(kb_id, page_size=200):
    all_docs = []
    page = 1
    while True:
        params = {
            "kb_id": kb_id,
            "page": page,
            "page_size": page_size,
        }
        data = {
            "tenant_id": tenant_id,
        }
        try:
            resp = requests.post(url_base + "list", params=params, json=data)
            data = resp.json()
        except Exception as e:
            print(f"请求异常 (page={page}): {e}")
            break

        if (
            not isinstance(data, dict)
            or "data" not in data
            or "docs" not in data["data"]
        ):
            print(f"返回异常 (page={page}): {data}")
            break

        docs = data["data"]["docs"]
        total = data["data"]["total"]

        if not docs:
            break

        all_docs.extend(docs)
        print(f"已取 {len(all_docs)}/{total} 个文档")

        if len(all_docs) >= total:
            break
        page += 1

    return all_docs


def clear_kb(kb_id, batch_size=200):
    docs = fetch_all_docs(kb_id)
    if not docs:
        print(f"知识库 {kb_id} 已为空，无需删除。")
        return

    doc_ids = [doc["id"] for doc in docs if "id" in doc]
    print(f"将删除 {len(doc_ids)} 个文档...")

    for i in tqdm(range(0, len(doc_ids), batch_size), desc="删除进度", unit="doc"):
        batch = doc_ids[i : i + batch_size]
        try:
            payload = {"doc_id": batch, "tenant_id": tenant_id}
            resp = requests.post(url_base + "rm", json=payload, timeout=30)
            if resp.status_code == 200 and resp.json().get("data", False):
                print(f"删除批次 {i // batch_size + 1} 成功")
            else:
                print(f"删除批次 {i // batch_size + 1} 失败 ({resp.status_code})")
            time.sleep(0.1)
        except Exception as e:
            print(f"批次 {i//batch_size+1} 删除异常: {e}")


def get_sys_name():
    return f"{os.getenv("USERNAME") or os.getenv("USER")}@{socket.gethostname()}"


def upload(name, isdir: bool = True):

    data = {"kb_id": kb_id, "tenant_id": tenant_id}

    if isdir:
        folder = name

        all_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ]

        # filter filename with `lmem_` prefix and `.json` suffix
        all_files = [
            f
            for f in all_files
            if os.path.basename(f).startswith("lmem_") and f.endswith(".json")
        ]
    else:
        all_files = [name]

    system_info = get_sys_name()

    for file in all_files:
        # insert system_info into file(json)
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            try:
                data_json = json.loads(content)
            except Exception as e:
                print(f"Error loading JSON from {file}: {e}")
                continue

            # Insert system_info into data_json
            data_json["system_info"] = system_info

            # Write the updated JSON back to tmpdirs without harm the file
            tmp_file = file.replace(".json", "_upload.json")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data_json, f, ensure_ascii=False, indent=2)

    batch_size = 20

    for i in range(0, 1, batch_size):
        batch = all_files[i : i + batch_size]

        files = [
            ("file", open(file.replace(".json", "_upload.json"), "rb"))
            for file in batch
        ]

        try:
            resp = requests.post(url_base + "upload", data=data, files=files)
            print(f"Batch {i // batch_size + 1}: {len(batch)} files")
            print("Status:", resp.status_code)
            try:
                print("Response:", resp.json())
            except Exception:
                print("Response text:", resp.text)
        finally:
            for _, f in files:
                f.close()

    # remove tmp files
    for file in all_files:
        tmp_file = file.replace(".json", "_upload.json")
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


from sys import argv
import shutil

if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python lmemuploader.py [upload|clear|upload_curr]")
        exit(1)

    command = argv[1]
    if command == "upload":
        upload("lmems", isdir=True)
    elif command == "clear":
        clear_kb(kb_id)
    elif command == "upload_curr":
        curdate = time.strftime("%Y%m%d%H%M%S", time.localtime())
        shutil.copy("lmems/lmem.json", f"lmems/lmem_{get_sys_name()}_{curdate}.json")
        upload(f"lmems/lmem_{get_sys_name()}_{curdate}.json", isdir=False)
        os.remove(f"lmems/lmem_{get_sys_name()}_{curdate}.json")
    else:
        print("Unknown command. Use 'upload', 'upload_curr' or 'clear'.")
