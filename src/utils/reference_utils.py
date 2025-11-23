import threading
import json
import hashlib
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = get_project_root()


class SessionMap:
    # 使用一个字典管理一个references map，key是int ID，value是reference对象，
    # 同时维护一个 _index: reference_key -> id，加速查询

    def __init__(self):
        self.reference_map = {}
        self._index = {}  # key(str) -> id(int)
        self.next_id = 1  # 从1开始分配ID
        self._lock = threading.RLock()  # 并发安全

    def _key_for_ref(self, reference):
        """为 reference 生成稳定的键：
        优先使用 reference['source'] 作为主键，但若同时存在 content 字段，则把 source 与 content 的哈希结合起来，
        这样即便 source 相同但内容不同也会得到不同的键/ID。
        若 reference 中既没有 source 也没有 content，则退回到对整个对象的 JSON 序列化或 repr，保证不会抛异常。
        """
        source = None
        content = None
        try:
            if isinstance(reference, dict):
                source = reference.get("source")
                content = reference.get("content")
            else:
                source = getattr(reference, "source", None)
                content = getattr(reference, "content", None)
        except Exception:
            source = None
            content = None

        # 如果存在 source，则优先使用 source 作为键的一部分
        if source is not None:
            # 如果同时存在 content，则使用 content 的哈希与 source 组合，避免相同 source 的不同内容被合并
            if content is not None:
                try:
                    # 尽量统一 content 的表示（若不是字符串，先序列化），再计算 sha256
                    if not isinstance(content, (str, bytes)):
                        content_bytes = json.dumps(
                            content, sort_keys=True, ensure_ascii=False
                        ).encode("utf-8")
                    else:
                        content_bytes = (
                            content.encode("utf-8")
                            if isinstance(content, str)
                            else content
                        )
                    h = hashlib.sha256(content_bytes).hexdigest()
                    return f"{source}::sha256:{h}"
                except Exception:
                    # 如果哈希计算失败，退回到将 content 的 repr 附加到 source
                    try:
                        return f"{source}::repr:{repr(content)}"
                    except Exception:
                        return str(source)
            # 只有 source，没有 content，则仅用 source
            return str(source)

        # 没有 source 时退回到对整个对象的 JSON 序列化或 repr
        try:
            s = json.dumps(
                reference, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
        except Exception:
            s = repr(reference)
        return s

    def add_references(self, references):
        """批量添加 references。对于已存在的 reference 返回已有的 id，新增分配新的 id。
        返回对应的 id 列表（与输入顺序一致）。
        """
        ids = []
        with self._lock:
            for ref in references:
                key = self._key_for_ref(ref)
                existing_id = self._index.get(key)
                if existing_id is not None:
                    ids.append(existing_id)
                else:
                    ref_id = self.next_id
                    self.reference_map[ref_id] = ref
                    self._index[key] = ref_id
                    ids.append(ref_id)
                    self.next_id += 1
        return ids

    def get_references(self, ids):
        """根据 id 列表返回 reference 对象列表，丢弃不存在的 id。"""
        refs = []
        with self._lock:
            for ref_id in ids:
                if ref_id in self.reference_map:
                    refs.append(self.reference_map[ref_id])
        return refs

    def get_reference_id(self, reference):
        """返回单个 reference 的 id（若不存在返回 None）。"""
        key = self._key_for_ref(reference)
        with self._lock:
            return self._index.get(key)

    def get_reference_ids(self, references):
        """批量获取 reference 对应的 id 列表，不存在时返回 None 于对应位置。"""
        ids = []
        with self._lock:
            for ref in references:
                ids.append(self._index.get(self._key_for_ref(ref)))
        return ids

    def remove_reference_by_id(self, ref_id):
        """可选：根据 id 删除 reference，同时更新索引。若 id 不存在返回 False。"""
        with self._lock:
            if ref_id not in self.reference_map:
                return False
            ref = self.reference_map.pop(ref_id)
            key = self._key_for_ref(ref)
            # 仅在索引仍指向该 id 时删除（防止键被复用的竞态情况）
            if self._index.get(key) == ref_id:
                del self._index[key]
            return True

    def save(self, file_path: str):
        """将 reference_map 保存到指定文件路径，格式为 JSON。"""
        with self._lock:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.reference_map, f, ensure_ascii=False, indent=2)


class ReferenceMap:
    ## 管理多个 session 的 references map，使用session_id作为区分
    def __init__(self):
        self._session_maps = {}
        self._lock = threading.RLock()  # 并发安全

    def get_session_map(self, session_id):
        """获取指定 session_id 的 SessionMap 实例，若不存在则创建。"""
        with self._lock:
            if session_id not in self._session_maps:
                self._session_maps[session_id] = SessionMap()
            return self._session_maps[session_id]

    def add_references(self, session_id, references):
        """向指定 session_id 添加 references，返回对应的 id 列表。"""
        session_map = self.get_session_map(session_id)
        return session_map.add_references(references)

    def get_references(self, session_id, ids):
        """根据 session_id 和 id 列表获取 references。"""
        session_map = self.get_session_map(session_id)
        return session_map.get_references(ids)

    def get_reference_ids(self, session_id, references):
        """根据 session_id 和 references 列表获取对应的 id 列表。"""
        session_map = self.get_session_map(session_id)
        return session_map.get_reference_ids(references)

    def get_reference_id(self, session_id, reference):
        """根据 session_id 和单个 reference 获取对应的 id。"""
        session_map = self.get_session_map(session_id)
        return session_map.get_reference_id(reference)

    def remove_reference_by_id(self, session_id, ref_id):
        """根据 session_id 和 id 删除对应的 reference。"""
        session_map = self.get_session_map(session_id)
        return session_map.remove_reference_by_id(ref_id)

    def save_session(self, session_id):
        """将指定 session_id 的 reference_map 保存到文件。"""
        file_path = PROJECT_ROOT / "references_logs" / f"{session_id}_references.json"
        session_map = self.get_session_map(session_id)
        session_map.save(file_path)

    def get_session_ref_map(self, session_id):
        file_path = PROJECT_ROOT / "references_logs" / f"{session_id}_references.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                reference_map = json.load(f)
            return reference_map
        session_map = self.get_session_map(session_id)
        return session_map.reference_map


global_reference_map = ReferenceMap()
