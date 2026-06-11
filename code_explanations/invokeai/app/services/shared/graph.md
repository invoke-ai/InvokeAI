# توثيق ملف: graph.py

## مسار الملف الأصلي
```
invokeai/app/services/shared/graph.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/services/shared/graph.py
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **نظام الرسم البياني** (Graph System) الذي يُشكّل الأساس لتنفيذ الأنابيب في InvokeAI. يحتوي على فئات `Graph` و`GraphExecutionState` اللتان تُديران الهيكل والتنفيذ للرسوم البيانية.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 مكتبات Python الأساسية
```python
import copy
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, Literal, Optional, Type, TypeVar, Union, get_args, get_origin
```

### 2.2 NetworkX
```python
import networkx as nx
```

### 2.3 Pydantic
```python
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, GetJsonSchemaHandler, PrivateAttr, ValidationError, field_validator
from pydantic.fields import Field
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
```

### 2.4 مكتبات المشروع
```python
from invokeai.app.invocations import *
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation, BaseInvocationOutput, InvocationRegistry, invocation, invocation_output,
)
from invokeai.app.invocations.fields import Input, InputField, OutputField, UIType
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import uuid_string
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 EdgeConnection
```python
class EdgeConnection(BaseModel):
    node_id: str = Field(description="The id of the node for this edge connection")
    field: str = Field(description="The field for this connection")

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and getattr(other, "node_id", None) == self.node_id
            and getattr(other, "field", None) == self.field
        )

    def __hash__(self):
        return hash(f"{self.node_id}.{self.field}")
```

### 3.2 Edge
```python
class Edge(BaseModel):
    source: EdgeConnection = Field(description="The connection for the edge's from node and field")
    destination: EdgeConnection = Field(description="The connection for the edge's to node and field")

    def __str__(self):
        return f"{self.source.node_id}.{self.source.field} -> {self.destination.node_id}.{self.destination.field}"
```

### 3.3 _PreparedExecRegistry
```python
class _PreparedExecRegistry:
    """Tracks prepared execution nodes and their relationship to source graph nodes."""

    def __init__(self, prepared_source_mapping: dict[str, str], source_prepared_mapping: dict[str, set[str]], metadata: dict[str, _PreparedExecNodeMetadata]) -> None:
        self._prepared_source_mapping = prepared_source_mapping
        self._source_prepared_mapping = source_prepared_mapping
        self._metadata = metadata

    def register(self, exec_node_id: str, source_node_id: str) -> None:
        self._prepared_source_mapping[exec_node_id] = source_node_id
        self._metadata[exec_node_id] = _PreparedExecNodeMetadata(source_node_id=source_node_id)
        if source_node_id not in self._source_prepared_mapping:
            self._source_prepared_mapping[source_node_id] = set()
        self._source_prepared_mapping[source_node_id].add(exec_node_id)

    def get_metadata(self, exec_node_id: str) -> _PreparedExecNodeMetadata:
        metadata = self._metadata.get(exec_node_id)
        if metadata is None:
            metadata = _PreparedExecNodeMetadata(source_node_id=self._prepared_source_mapping[exec_node_id])
            self._metadata[exec_node_id] = metadata
        return metadata

    def get_source_node_id(self, exec_node_id: str) -> str:
        metadata = self._metadata.get(exec_node_id)
        if metadata is not None:
            return metadata.source_node_id
        return self._prepared_source_mapping[exec_node_id]

    def get_prepared_ids(self, source_node_id: str) -> set[str]:
        return self._source_prepared_mapping.get(source_node_id, set())

    def set_state(self, exec_node_id: str, state: PreparedExecState) -> None:
        self.get_metadata(exec_node_id).state = state

    def get_iteration_path(self, exec_node_id: str) -> Optional[tuple[int, ...]]:
        metadata = self._metadata.get(exec_node_id)
        return metadata.iteration_path if metadata is not None else None

    def set_iteration_path(self, exec_node_id: str, iteration_path: tuple[int, ...]) -> None:
        self.get_metadata(exec_node_id).iteration_path = iteration_path
```

### 3.4 _IfBranchScheduler
```python
class _IfBranchScheduler:
    """Applies lazy `If` semantics by deferring, releasing, and skipping branch-local exec nodes."""

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state

    def _get_branch_input_sources(self, if_node_id: str, branch_field: str) -> set[str]:
        return {e.source.node_id for e in self._state.graph._get_input_edges(if_node_id, branch_field)}

    def _expand_with_ancestors(self, node_ids: set[str]) -> set[str]:
        expanded = set(node_ids)
        source_graph = self._state.graph.nx_graph_flat()
        for node_id in list(expanded):
            expanded.update(nx.ancestors(source_graph, node_id))
        return expanded

    def _node_outputs_stay_in_branch(self, node_id: str, if_node_id: str, branch_field: str, branch_nodes: set[str]) -> bool:
        output_edges = self._state.graph._get_output_edges(node_id)
        return all(
            edge.destination.node_id in branch_nodes
            or (edge.destination.node_id == if_node_id and edge.destination.field == branch_field)
            for edge in output_edges
        )

    def _prune_nonexclusive_branch_nodes(self, if_node_id: str, branch_field: str, candidate_nodes: set[str]) -> set[str]:
        exclusive_nodes = set(candidate_nodes)
        changed = True
        while changed:
            changed = False
            for node_id in list(exclusive_nodes):
                if self._node_outputs_stay_in_branch(node_id, if_node_id, branch_field, exclusive_nodes):
                    continue
                exclusive_nodes.remove(node_id)
                changed = True
        return exclusive_nodes

    def _get_matching_prepared_if_ids(self, if_node_id: str, iteration_path: tuple[int, ...]) -> list[str]:
        prepared_if_ids = self._state._prepared_registry().get_prepared_ids(if_node_id)
        return [pid for pid in prepared_if_ids if self._state._get_iteration_path(pid) == iteration_path]

    def _has_unresolved_matching_if(self, if_node_id: str, iteration_path: tuple[int, ...]) -> bool:
        matching_prepared_if_ids = self._get_matching_prepared_if_ids(if_node_id, iteration_path)
        if not matching_prepared_if_ids:
            return True
        return not all(pid in self._state._resolved_if_exec_branches for pid in matching_prepared_if_ids)

    def _apply_condition_inputs(self, exec_node_id: str, node: IfInvocation) -> bool:
        return self._state._apply_if_condition_inputs(exec_node_id, node)

    def _get_selected_branch_fields(self, node: IfInvocation) -> tuple[str, str]:
        selected_field = "true_input" if node.condition else "false_input"
        unselected_field = "false_input" if node.condition else "true_input"
        return selected_field, unselected_field

    def _prune_unselected_if_inputs(self, exec_node_id: str, unselected_field: str) -> None:
        for edge in self._state.execution_graph._get_input_edges(exec_node_id, unselected_field):
            if edge.source.node_id not in self._state.executed:
                if self._state.indegree[exec_node_id] == 0:
                    raise RuntimeError(f"indegree underflow for {exec_node_id} when pruning {unselected_field}")
                self._state.indegree[exec_node_id] -= 1
            self._state.execution_graph.delete_edge(edge)
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من وجود العقد
```python
try:
    node = self.graph.get_node(batch_data.node_path)
except NodeNotFoundError:
    raise NodeNotFoundError(f"Node {batch_data.node_path} not found in graph")
```

### 4.2 التعامل مع التكرار
```python
def _has_unresolved_matching_if(self, if_node_id: str, iteration_path: tuple[int, ...]) -> bool:
    matching_prepared_if_ids = self._get_matching_prepared_if_ids(if_node_id, iteration_path)
    if not matching_prepared_if_ids:
        return True
    return not all(pid in self._state._resolved_if_exec_branches for pid in matching_prepared_if_ids)
```

### 4.3 التعامل مع الأخطاء
```python
if self._state.indegree[exec_node_id] == 0:
    raise RuntimeError(f"indegree underflow for {exec_node_id} when pruning {unselected_field}")
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **كفاءة NetworkX**: استخدام مكتبة NetworkX للتعامل مع الرسوم البيانية المعقدة.
2. **دعم التكرار**: دعم التكرار المتعدد للرسوم البيانية.
3. **إدارة الذاكرة**: استخدام الـ registries لتتبع العقد المُعدة.

### نقاط الضعف
1. **تعقيد الكود**: معقد نسبياً للفهم.
2. **استهلاك الذاكرة**: قد يستهلك ذاكرة كبيرة للرسوم البيانية الكبيرة.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Graph Execution Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Graph                                                      │
│       │                                                     │
│       ├── nodes: dict[str, BaseInvocation]                  │
│       ├── edges: list[Edge]                                 │
│       └── nx_graph_flat()                                   │
│       │                                                     │
│       ▼                                                     │
│  GraphExecutionState                                        │
│       │                                                     │
│       ├── graph: Graph                                      │
│       ├── execution_graph: Graph                            │
│       ├── executed: set[str]                                │
│       ├── indegree: dict[str, int]                          │
│       └── _prepared_registry(): _PreparedExecRegistry       │
│       │                                                     │
│       ▼                                                     │
│  _IfBranchScheduler                                         │
│       │                                                     │
│       ├── _get_branch_input_sources()                       │
│       ├── _expand_with_ancestors()                          │
│       ├── _prune_nonexclusive_branch_nodes()                │
│       └── _apply_branch_resolution()                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [NetworkX](https://networkx.org/)
- [Graph Theory](https://en.wikipedia.org/wiki/Graph_theory)
- [Directed Acyclic Graphs](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
