import json
HEAD_MASK = "HEAD_QUERY"
class Node:
    def __init__(self, node_id, question, related_chunks):
        self.node_id = node_id
        self.question = question
        self.related_chunks = related_chunks
        self.answer = None

    def __str__(self):
        if self.node_id == HEAD_MASK:
            return f"{self.node_id}: {self.question}\n {(self.node_id).replace(HEAD_MASK,'A')}: {self.answer}"
        else:
            return f"{self.node_id}: {self.question}\n {(self.node_id).replace('Q','A')}: {self.answer}"

    def set_answer(self, answer):
        self.answer = answer

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.head = None

    def add_node(self, node):
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node
            self.edges[node.node_id] = []

    def add_edge(self, from_node_id, to_node_id):
        if from_node_id in self.nodes and to_node_id in self.nodes:
            self.edges[from_node_id].append(to_node_id)

    def topological_sort_util(self, node_id, visited, stack):
        visited[node_id] = True
        for neighbor in self.edges[node_id]:
            if not visited[neighbor]:
                self.topological_sort_util(neighbor, visited, stack)
        stack.insert(0, node_id)

    def topological_sort(self):
        visited = {node_id: False for node_id in self.nodes}
        stack = []
        for node_id in self.nodes:
            if not visited[node_id]:
                self.topological_sort_util(node_id, visited, stack)
        stack = [node_id for node_id in stack if node_id != HEAD_MASK]
        return stack

    def __str__(self):
        sorted_nodes = self.topological_sort()
        return "\n".join(str(self.nodes[node_id]) for node_id in sorted_nodes)
    
    def load_dag_from_json(self, data):
        for parent_query, child_query in data["DAG"]:
            # Extract ID and question for parent query
            parent_id, parent_question = self.extract_id_and_question(parent_query)
            # Extract ID and question for child query
            child_id, child_question = self.extract_id_and_question(child_query)

            if parent_id not in self.nodes:
                self.add_node(Node(parent_id, parent_question, []))
            if child_id not in self.nodes:
                self.add_node(Node(child_id, child_question, []))
            self.add_edge(parent_id, child_id)

    def extract_id_and_question(self, query):
        if isinstance(query,list):
            query = query[0]
        parts = query.split(": ", 1)
        if len(parts) == 2:
            if self.head is None:
                self.head = parts[0]
                return HEAD_MASK, parts[1]
            else:
                if parts[0] == self.head:
                    return HEAD_MASK, parts[1]
                else:
                    return parts[0], parts[1]
        else:
            tmp_id = str(hash(query))
            if self.head is None:
                self.head = tmp_id
                return HEAD_MASK, query
            else:
                if tmp_id == self.head:
                    return HEAD_MASK, query
                else:
                    return tmp_id, query
    
    def get_head(self):
        return self.nodes[HEAD_MASK]