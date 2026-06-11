# توثيق ملف: session_queue.py

## مسار الملف الأصلي
```
invokeai/app/api/routers/session_queue.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/api/routers/session_queue.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **محور الطابور** (Queue Router) الذي يوفر واجهة برمجة تطبيقات REST لإدارة طابور الجلسات. وهو مسؤول عن إضافة الدفعات، وإدارة العناصر، ومراقبة الحالة.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 FastAPI
```python
from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter
```

### 2.2 Pydantic
```python
from pydantic import BaseModel
```

### 2.3 مكتبات المشروع
```python
from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.session_queue.session_queue_common import (
    Batch, BatchStatus, SessionQueueItem, SessionQueueStatus, ...
)
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 تعريف المحور
```python
session_queue_router = APIRouter(prefix="/v1/queue", tags=["queue"])
```

### 3.2 نموذج الحالة المشتركة
```python
class SessionQueueAndProcessorStatus(BaseModel):
    queue: SessionQueueStatus
    processor: SessionProcessorStatus
```

### 3.3 دالة التنظيف
```python
def sanitize_queue_item_for_user(
    queue_item: SessionQueueItem, current_user_id: str, is_admin: bool
) -> SessionQueueItem:
    if is_admin or queue_item.user_id == current_user_id:
        return queue_item

    sanitized_item = queue_item.model_copy(deep=False)
    sanitized_item.user_id = "redacted"
    sanitized_item.batch_id = "redacted"
    sanitized_item.session_id = "redacted"
    sanitized_item.session = GraphExecutionState(id="redacted", graph=Graph())
    return sanitized_item
```

### 3.4 نقاط النهاية (Endpoints)

#### enqueue_batch
```python
@session_queue_router.post("/{queue_id}/enqueue_batch", operation_id="enqueue_batch")
async def enqueue_batch(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    batch: Batch = Body(description="Batch to process"),
    prepend: bool = Body(default=False, description="Whether or not to prepend this batch"),
) -> EnqueueBatchResult:
    return await ApiDependencies.invoker.services.session_queue.enqueue_batch(
        queue_id=queue_id, batch=batch, prepend=prepend, user_id=current_user.user_id
    )
```

#### list_all_queue_items
```python
@session_queue_router.get("/{queue_id}/list_all", operation_id="list_all_queue_items")
async def list_all_queue_items(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    destination: Optional[str] = Query(default=None, description="The destination of queue items to fetch"),
) -> list[SessionQueueItem]:
    items = ApiDependencies.invoker.services.session_queue.list_all_queue_items(queue_id=queue_id, destination=destination)
    return [sanitize_queue_item_for_user(item, current_user.user_id, current_user.is_admin) for item in items]
```

#### get_queue_item_ids
```python
@session_queue_router.get("/{queue_id}/item_ids", operation_id="get_queue_item_ids")
async def get_queue_item_ids(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
) -> ItemIdsResult:
    user_id = None if current_user.is_admin else current_user.user_id
    return ApiDependencies.invoker.services.session_queue.get_queue_item_ids(queue_id=queue_id, order_dir=order_dir, user_id=user_id)
```

#### resume / pause
```python
@session_queue_router.put("/{queue_id}/processor/resume", operation_id="resume")
async def resume(current_user: AdminUserOrDefault, queue_id: str = Path(...)) -> SessionProcessorStatus:
    return ApiDependencies.invoker.services.session_processor.resume()

@session_queue_router.put("/{queue_id}/processor/pause", operation_id="pause")
async def pause(current_user: AdminUserOrDefault, queue_id: str = Path(...)) -> SessionProcessorStatus:
    return ApiDependencies.invoker.services.session_processor.pause()
```

#### cancel_all_except_current
```python
@session_queue_router.put("/{queue_id}/cancel_all_except_current", operation_id="cancel_all_except_current")
async def cancel_all_except_current(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> CancelAllExceptCurrentResult:
    user_id = None if current_user.is_admin else current_user.user_id
    return ApiDependencies.invoker.services.session_queue.cancel_all_except_current(queue_id=queue_id, user_id=user_id)
```

#### clear
```python
@session_queue_router.put("/{queue_id}/clear", operation_id="clear")
async def clear(current_user: CurrentUserOrDefault, queue_id: str = Path(...)) -> ClearResult:
    queue_item = ApiDependencies.invoker.services.session_queue.get_current(queue_id)
    if queue_item is not None:
        if queue_item.user_id != current_user.user_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="You do not have permission to cancel the currently executing queue item")
        ApiDependencies.invoker.services.session_queue.cancel_queue_item(queue_item.item_id)
    user_id = None if current_user.is_admin else current_user.user_id
    clear_result = ApiDependencies.invoker.services.session_queue.clear(queue_id, user_id=user_id)
    return clear_result
```

#### retry_items_by_id
```python
@session_queue_router.put("/{queue_id}/retry_items_by_id", operation_id="retry_items_by_id")
async def retry_items_by_id(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_ids: list[int] = Body(description="The queue item ids to retry"),
) -> RetryItemsResult:
    if not current_user.is_admin:
        for item_id in item_ids:
            try:
                queue_item = ApiDependencies.invoker.services.session_queue.get_queue_item(item_id)
                if queue_item.user_id != current_user.user_id:
                    raise HTTPException(status_code=403, detail=f"You do not have permission to retry queue item {item_id}")
            except SessionQueueItemNotFoundError:
                continue
    return ApiDependencies.invoker.services.session_queue.retry_items_by_id(queue_id=queue_id, item_ids=item_ids)
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من الصلاحيات
```python
if not current_user.is_admin:
    for item_id in item_ids:
        queue_item = ApiDependencies.invoker.services.session_queue.get_queue_item(item_id)
        if queue_item.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail=f"You do not have permission to retry queue item {item_id}")
```

### 4.2 تنظيف البيانات
```python
return [sanitize_queue_item_for_user(item, current_user.user_id, current_user.is_admin) for item in items]
```

### 4.3 معالجة الأخطاء
```python
try:
    return await ApiDependencies.invoker.services.session_queue.enqueue_batch(...)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Unexpected error while enqueuing batch: {e}")
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **صلاحيات متعددة المستخدمين**: دعم كامل لمستخدمين متعددين مع صلاحيات مختلفة.
2. **تنظيف البيانات**: حماية بيانات المستخدمين الآخرين.
3. **توثيق شامل**: كل نقطة نهاية موثقة بوضوح.
4. **معالجة أخطاء**: معالجة شاملة للأخطاء.

### نقاط الضعف
1. **عدد كبير من نقاط النهاية**: قد يصعب الصيانة.
2. **تكرار الكود**: بعض الأكواد مكررة في نقاط النهاية المختلفة.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Queue Router API Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  POST /{queue_id}/enqueue_batch                             │
│       └── Add batch to queue                                │
│                                                             │
│  GET /{queue_id}/list_all                                   │
│       └── List all queue items (sanitized)                  │
│                                                             │
│  GET /{queue_id}/item_ids                                   │
│       └── Get queue item IDs                                │
│                                                             │
│  PUT /{queue_id}/processor/resume                           │
│       └── Resume processor (admin only)                     │
│                                                             │
│  PUT /{queue_id}/processor/pause                            │
│       └── Pause processor (admin only)                      │
│                                                             │
│  PUT /{queue_id}/cancel_all_except_current                  │
│       └── Cancel all items except current                   │
│                                                             │
│  PUT /{queue_id}/clear                                      │
│       └── Clear all items                                   │
│                                                             │
│  PUT /{queue_id}/retry_items_by_id                          │
│       └── Retry failed items                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [FastAPI Routers](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [REST API Design](https://restfulapi.net/)
- [Queue Management](https://en.wikipedia.org/wiki/Job_queue)
