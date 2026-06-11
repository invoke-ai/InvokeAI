# توثيق ملف: session_queue_common.py

## مسار الملف الأصلي
```
invokeai/app/services/session_queue/session_queue_common.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/services/session_queue/session_queue_common.py
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **المكونات المشتركة** لطابور الجلسات في InvokeAI. يحتوي على نماذج البيانات والوظائف المساعدة المستخدمة في إدارة الطابور.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 مكتبات Python الأساسية
```python
import datetime
import json
from itertools import chain, product
from typing import Generator, Literal, Optional, TypeAlias, Union
```

### 2.2 Pydantic
```python
from pydantic import (
    AliasChoices, BaseModel, ConfigDict, Field, StrictStr,
    TypeAdapter, field_validator, model_validator,
)
from pydantic_core import to_jsonable_python
```

### 2.3 مكتبات المشروع
```python
from invokeai.app.invocations.fields import ImageField
from invokeai.app.services.shared.graph import Graph, GraphExecutionState, NodeNotFoundError
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID
from invokeai.app.util.misc import uuid_string
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 أخطاء الدفعية
```python
class BatchZippedLengthError(ValueError):
    """Raise when a batch has items of different lengths."""

class BatchItemsTypeError(ValueError):
    """Raise when a batch has items of different types."""

class BatchDuplicateNodeFieldError(ValueError):
    """Raise when a batch has duplicate node_path and field_name."""

class TooManySessionsError(ValueError):
    """Raise when too many sessions are requested."""

class SessionQueueItemNotFoundError(ValueError):
    """Raise when a queue item is not found."""
```

### 3.2 نماذج البيانات الأساسية

#### NodeFieldValue
```python
class NodeFieldValue(BaseModel):
    node_path: str = Field(description="The node into which this batch data item will be substituted.")
    field_name: str = Field(description="The field into which this batch data item will be substituted.")
    value: BatchDataType = Field(description="The value to substitute into the node/field.")
```

#### BatchDatum
```python
class BatchDatum(BaseModel):
    node_path: str = Field(description="The node into which this batch data collection will be substituted.")
    field_name: str = Field(description="The field into which this batch data collection will be substituted.")
    items: list[BatchDataType] = Field(default_factory=list, description="The list of items to substitute into the node/field.")
```

#### Batch
```python
class Batch(BaseModel):
    batch_id: str = Field(default_factory=uuid_string, description="The ID of the batch")
    origin: str | None = Field(default=None, description="The origin of this queue item.")
    destination: str | None = Field(default=None, description="The destination of this queue item.")
    data: Optional[BatchDataCollection] = Field(default=None, description="The batch data collection.")
    graph: Graph = Field(description="The graph to initialize the session with")
    workflow: Optional[WorkflowWithoutID] = Field(default=None, description="The workflow to initialize the session with")
    runs: int = Field(default=1, ge=1, description="Int stating how many times to iterate through all possible batch indices")
```

### 3.3 التحقق من صحة البيانات

#### التحقق من الأطوال
```python
@field_validator("data")
def validate_lengths(cls, v: Optional[BatchDataCollection]):
    if v is None:
        return v
    for batch_data_list in v:
        first_item_length = len(batch_data_list[0].items) if batch_data_list and batch_data_list[0].items else 0
        for i in batch_data_list:
            if len(i.items) != first_item_length:
                raise BatchZippedLengthError("Zipped batch items must all have the same length")
    return v
```

#### التحقق من الأنواع
```python
@field_validator("data")
def validate_types(cls, v: Optional[BatchDataCollection]):
    if v is None:
        return v
    for batch_data_list in v:
        for datum in batch_data_list:
            if not datum.items:
                continue
            if all(isinstance(item, (int, float)) for item in datum.items):
                continue
            first_item_type = type(datum.items[0])
            for item in datum.items:
                if type(item) is not first_item_type:
                    raise BatchItemsTypeError("All items in a batch must have the same type")
    return v
```

#### التحقق من تفرد تعيينات الحقول
```python
@field_validator("data")
def validate_unique_field_mappings(cls, v: Optional[BatchDataCollection]):
    if v is None:
        return v
    paths: set[tuple[str, str]] = set()
    for batch_data_list in v:
        for datum in batch_data_list:
            pair = (datum.node_path, datum.field_name)
            if pair in paths:
                raise BatchDuplicateNodeFieldError("Each batch data must have unique node_id and field_name")
            paths.add(pair)
    return v
```

### 3.4 SessionQueueItem
```python
class SessionQueueItem(BaseModel):
    """Session queue item without the full graph. Used for serialization."""

    item_id: int = Field(description="The identifier of the session queue item")
    status: QUEUE_ITEM_STATUS = Field(default="pending", description="The status of this queue item")
    status_sequence: int | None = Field(default=None, description="A monotonically increasing version for this queue item's visible status lifecycle")
    priority: int = Field(default=0, description="The priority of this queue item")
    batch_id: str = Field(description="The ID of the batch associated with this queue item")
    origin: str | None = Field(default=None, description="The origin of this queue item.")
    destination: str | None = Field(default=None, description="The destination of this queue item.")
    session_id: str = Field(description="The ID of the session associated with this queue item.")
    error_type: Optional[str] = Field(default=None, description="The error type if this queue item errored")
    error_message: Optional[str] = Field(default=None, description="The error message if this queue item errored")
    error_traceback: Optional[str] = Field(default=None, description="The error traceback if this queue item errored")
    created_at: Union[datetime.datetime, str] = Field(description="When this queue item was created")
    updated_at: Union[datetime.datetime, str] = Field(description="When this queue item was updated")
    started_at: Optional[Union[datetime.datetime, str]] = Field(description="When this queue item was started")
    completed_at: Optional[Union[datetime.datetime, str]] = Field(description="When this queue item was completed")
    queue_id: str = Field(description="The id of the queue with which this item is associated")
    user_id: str = Field(default="system", description="The id of the user who created this queue item")
    user_display_name: Optional[str] = Field(default=None, description="The display name of the user")
    user_email: Optional[str] = Field(default=None, description="The email of the user")
    field_values: Optional[list[NodeFieldValue]] = Field(default=None, description="The field values that were used for this queue item")
    retried_from_item_id: Optional[int] = Field(default=None, description="The item_id of the queue item that this item was retried from")
    session: GraphExecutionState = Field(description="The fully-populated session to be executed")
    workflow: Optional[WorkflowWithoutID] = Field(default=None, description="The workflow associated with this queue item")
```

### 3.5 نماذج نتائج الاستعلام

#### SessionQueueStatus
```python
class SessionQueueStatus(BaseModel):
    queue_id: str = Field(..., description="The ID of the queue")
    item_id: Optional[int] = Field(description="The current queue item id")
    batch_id: Optional[str] = Field(description="The current queue item's batch id")
    session_id: Optional[str] = Field(description="The current queue item's session id")
    pending: int = Field(..., description="Number of queue items with status 'pending'")
    in_progress: int = Field(..., description="Number of queue items with status 'in_progress'")
    completed: int = Field(..., description="Number of queue items with status 'complete'")
    failed: int = Field(..., description="Number of queue items with status 'error'")
    canceled: int = Field(..., description="Number of queue items with status 'canceled'")
    total: int = Field(..., description="Total number of queue items")
```

### 3.6 الدوال المساعدة

#### إنشاء جلسات الدفعية
```python
def create_session_nfv_tuples(batch: Batch, maximum: int) -> Generator[tuple[str, str, str], None, None]:
    """
    Given a batch and a maximum number of sessions to create, generate a tuple of session_id, session_json, and
    field_values_json for each session.
    """
    data: list[list[tuple[dict]]] = []
    batch_data_collection = batch.data if batch.data is not None else []

    for batch_datum_list in batch_data_collection:
        node_field_values_to_zip: list[list[dict]] = []
        for batch_datum in batch_datum_list:
            node_field_values = [
                {"node_path": batch_datum.node_path, "field_name": batch_datum.field_name, "value": item}
                for item in batch_datum.items
            ]
            node_field_values_to_zip.append(node_field_values)
        data.append(list(zip(*node_field_values_to_zip, strict=True)))

    graph_as_dict = batch.graph.model_dump(warnings=False, exclude_none=True)
    session_dict = GraphExecutionState(graph=Graph()).model_dump(warnings=False, exclude_none=True)

    count = 0
    for _ in range(batch.runs):
        for d in product(*data):
            if count >= maximum:
                return
            flat_node_field_values = list(chain.from_iterable(d))
            session_id = uuid_string()
            session_dict["id"] = session_id
            for nfv in flat_node_field_values:
                graph_as_dict["nodes"][nfv["node_path"]][nfv["field_name"]] = nfv["value"]
            session_dict["graph"] = graph_as_dict
            session_json = json.dumps(session_dict, default=to_jsonable_python)
            field_values_json = json.dumps(flat_node_field_values, default=to_jsonable_python)
            yield (session_id, session_json, field_values_json)
            count += 1
```

#### حساب عدد الجلسات
```python
def calc_session_count(batch: Batch) -> int:
    """Calculates the number of sessions that would be created by the batch."""
    if not batch.data:
        return batch.runs
    data = []
    for batch_datum_list in batch.data:
        to_zip = []
        for batch_datum in batch_datum_list:
            batch_data_items = range(len(batch_datum.items))
            to_zip.append(batch_data_items)
        data.append(list(zip(*to_zip, strict=True)))
    data_product = list(product(*data))
    return len(data_product) * batch.runs
```

#### تحضير القيم للإدخال
```python
def prepare_values_to_insert(queue_id: str, batch: Batch, priority: int, max_new_queue_items: int, user_id: str = "system") -> list[ValueToInsertTuple]:
    """Given a batch, prepare the values to insert into the session queue table."""
    values_to_insert: list[ValueToInsertTuple] = []
    workflow_json = json.dumps(batch.workflow, default=to_jsonable_python) if batch.workflow else None

    for session_id, session_json, field_values_json in create_session_nfv_tuples(batch, max_new_queue_items):
        values_to_insert.append((
            queue_id, session_json, session_id, batch.batch_id,
            field_values_json, priority, workflow_json,
            batch.origin, batch.destination, None, user_id,
        ))
    return values_to_insert
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من الأطوال المتفاوتة
```python
@field_validator("data")
def validate_lengths(cls, v: Optional[BatchDataCollection]):
    if v is None:
        return v
    for batch_data_list in v:
        first_item_length = len(batch_data_list[0].items) if batch_data_list and batch_data_list[0].items else 0
        for i in batch_data_list:
            if len(i.items) != first_item_length:
                raise BatchZippedLengthError("Zipped batch items must all have the same length")
    return v
```

### 4.2 التحقق من تطابق العقد
```python
@model_validator(mode="after")
def validate_batch_nodes_and_edges(self):
    if self.data is None:
        return self
    for batch_data_list in self.data:
        for batch_data in batch_data_list:
            try:
                node = self.graph.get_node(batch_data.node_path)
            except NodeNotFoundError:
                raise NodeNotFoundError(f"Node {batch_data.node_path} not found in graph")
            if batch_data.field_name not in type(node).model_fields:
                raise NodeNotFoundError(f"Field {batch_data.field_name} not found in node {batch_data.node_path}")
    return self
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **كفاءة الأداء**: استخدام `itertools.product` للحصول على جميع الت.tabulations.
2. **ذاكرة فعالة**: استخدام المولدات (generators) لتوفير الذاكرة.
3. **تشفير فعال**: استخدام `to_jsonable_python` للتعامل مع أنواع البيانات المعقدة.

### نقاط الضعف
1. **تعقيد الكود**: معقد نسبياً للفهم.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Session Queue Common Architecture              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Batch                                                      │
│       │                                                     │
│       ├── batch_id: str                                     │
│       ├── graph: Graph                                      │
│       ├── data: BatchDataCollection                         │
│       │     ├── BatchDatum                                  │
│       │     │     ├── node_path: str                        │
│       │     │     ├── field_name: str                       │
│       │     │     └── items: list[BatchDataType]            │
│       └── runs: int                                         │
│       │                                                     │
│       ▼                                                     │
│  create_session_nfv_tuples()                                │
│       │                                                     │
│       ├── Expand BatchDatum objects                         │
│       ├── Zip together                                     │
│       ├── Cartesian product                                │
│       └── Yield session tuples                              │
│       │                                                     │
│       ▼                                                     │
│  SessionQueueItem                                           │
│       │                                                     │
│       ├── item_id: int                                      │
│       ├── status: QUEUE_ITEM_STATUS                         │
│       ├── session: GraphExecutionState                      │
│       └── field_values: list[NodeFieldValue]                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/)
- [Cartesian Product](https://en.wikipedia.org/wiki/Cartesian_product)
- [Generator Patterns](https://wiki.python.org/moin/Generators)
