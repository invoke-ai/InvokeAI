# توثيق ملف: session_processor_default.py

## مسار الملف الأصلي
```
invokeai/app/services/session_processor/session_processor_default.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/services/session_processor/session_processor_default.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **معالج الجلسات الافتراضي** (Default Session Processor) الذي مسؤول عن تنفيذ سير عمل العقد (Node Graphs). وهو يدير دورة حياة الجلسة كاملة، بدءاً من جلب العناصر من الطابور وانتهاءً بحفظ النتائج.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 المكتبات المعيارية
```python
import gc
import traceback
from contextlib import suppress
from threading import BoundedSemaphore, Thread
from threading import Event as ThreadEvent
from typing import Optional
```
- **gc**: جامع القمامة لتقليل استهلاك الذاكرة.
- **traceback**: لتنسيق تتبع الأخطاء.
- **Thread/ThreadEvent**: للتعامل مع الخيوط المتعددة.

### 2.2 مكتبات المشروع
```python
from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.services.events.events_common import (
    BatchEnqueuedEvent, FastAPIEvent, QueueClearedEvent,
    QueueItemStatusChangedEvent, register_events
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_processor.session_processor_base import (
    SessionProcessorBase, SessionRunnerBase, OnBeforeRunSession,
    OnAfterRunSession, OnBeforeRunNode, OnAfterRunNode,
    OnNodeError, OnNonFatalProcessorError, InvocationServices
)
from invokeai.app.services.session_processor.session_processor_common import CanceledException, SessionProcessorStatus
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem, SessionQueueItemNotFoundError
from invokeai.app.services.shared.graph import NodeInputError
from invokeai.app.services.shared.invocation_context import InvocationContextData, build_invocation_context
from invokeai.app.util.profiler import Profiler
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 فئة DefaultSessionRunner

#### التهيئة
```python
class DefaultSessionRunner(SessionRunnerBase):
    def __init__(
        self,
        on_before_run_session_callbacks: Optional[list[OnBeforeRunSession]] = None,
        on_before_run_node_callbacks: Optional[list[OnBeforeRunNode]] = None,
        on_after_run_node_callbacks: Optional[list[OnAfterRunNode]] = None,
        on_node_error_callbacks: Optional[list[OnNodeError]] = None,
        on_after_run_session_callbacks: Optional[list[OnAfterRunSession]] = None,
    ):
```

#### تنفيذ الجلسة
```python
def run(self, queue_item: SessionQueueItem):
    self._on_before_run_session(queue_item=queue_item)

    while True:
        try:
            invocation = queue_item.session.next()
        except NodeInputError as e:
            self._on_node_error(invocation=e.node, queue_item=queue_item, ...)
            break

        if invocation is None or self._is_canceled():
            break

        self.run_node(invocation, queue_item)

        if queue_item.session.is_complete() or self._is_canceled():
            break

    self._on_after_run_session(queue_item=queue_item)
```

#### تنفيذ العقدة
```python
def run_node(self, invocation: BaseInvocation, queue_item: SessionQueueItem):
    try:
        with self._services.performance_statistics.collect_stats(invocation, queue_item.session_id):
            self._on_before_run_node(invocation, queue_item)

            data = InvocationContextData(
                invocation=invocation,
                source_invocation_id=queue_item.session.prepared_source_mapping[invocation.id],
                queue_item=queue_item,
            )
            context = build_invocation_context(data=data, services=self._services, is_canceled=self._is_canceled)

            output = invocation.invoke_internal(context=context, services=self._services)
            queue_item.session.complete(invocation.id, output)

            self._on_after_run_node(invocation, queue_item, output)

    except CanceledException:
        pass
    except Exception as e:
        self._on_node_error(invocation=invocation, queue_item=queue_item, ...)
```

### 3.2 فئة DefaultSessionProcessor

#### بدء التشغيل
```python
def start(self, invoker: Invoker) -> None:
    self._invoker = invoker
    self._resume_event = ThreadEvent()
    self._stop_event = ThreadEvent()
    self._poll_now_event = ThreadEvent()
    self._cancel_event = ThreadEvent()

    register_events(QueueClearedEvent, self._on_queue_cleared)
    register_events(BatchEnqueuedEvent, self._on_batch_enqueued)
    register_events(QueueItemStatusChangedEvent, self._on_queue_item_status_changed)

    self._thread = Thread(name="session_processor", target=self._process, daemon=True)
    self._thread.start()
```

#### معالجة الطابور
```python
def _process(self, stop_event, poll_now_event, resume_event, cancel_event):
    while not stop_event.is_set():
        poll_now_event.clear()
        resume_event.wait()

        self._queue_item = self._invoker.services.session_queue.dequeue()

        if self._queue_item is None:
            poll_now_event.wait(self._polling_interval)
            continue

        gc.collect()

        self.session_runner.run(queue_item=self._queue_item)
```

#### التعامل مع الإلغاء
```python
async def _on_queue_item_status_changed(self, event: FastAPIEvent[QueueItemStatusChangedEvent]):
    if self._queue_item and self._queue_item.item_id != event[1].item_id:
        return
    if self._queue_item and event[1].status in ["completed", "failed", "canceled"]:
        if event[1].status == "canceled":
            self._cancel_event.set()
        self._poll_now()
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 أخطاء العقدة
```python
except NodeInputError as e:
    error_type = e.__class__.__name__
    error_message = str(e)
    error_traceback = traceback.format_exc()
    self._on_node_error(invocation=e.node, queue_item=queue_item, ...)
```

### 4.2 الإلغاء
```python
except CanceledException:
    pass
```

### 4.3 أخطاء المعالج غير القاتلة
```python
def _on_non_fatal_processor_error(self, queue_item, error_type, error_message, error_traceback):
    self._invoker.services.logger.error(f"Non-fatal error in session processor {error_type}: {error_message}")
    if queue_item is not None:
        queue_item = self._invoker.services.session_queue.set_queue_item_session(queue_item.item_id, queue_item.session)
        queue_item = self._invoker.services.session_queue.fail_queue_item(...)
```

### 4.4 تنظيف الذاكرة
```python
gc.collect()
```
- يتم استدعاء `gc.collect()` قبل كل عقدة لتقليل استهلاك الذاكرة.

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **خيط منفصل**: المعالج يعمل في خيط منفصل لتجنب تجميد الواجهة.
2. **إلغاء رشيق**: دعم الإلغاء أثناء التنفيذ.
3. **جمع القمامة**: تنظيف الذاكرة بانتظام.
4. **إحصائيات الأداء**: تتبع أداء كل عقدة.

### نقاط الضعف
1. **خيط واحد**: استخدام خيط واحد فقط لمعالجة الجلسات.
2. **تعقيد الإلغاء**: التعامل مع الإلغاء معقد نسبياً.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Session Processor Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  start(invoker)                                             │
│       │                                                     │
│       ├── Create events (resume, stop, poll_now, cancel)    │
│       │                                                     │
│       ├── Register queue event handlers                     │
│       │                                                     │
│       └── Start processing thread                           │
│                                                             │
│  _process() [Thread]                                        │
│       │                                                     │
│       ├── while not stop_event.is_set():                    │
│       │     │                                               │
│       │     ├── resume_event.wait()                         │
│       │     │                                               │
│       │     ├── dequeue()                                   │
│       │     │                                               │
│       │     ├── gc.collect()                                │
│       │     │                                               │
│       │     └── session_runner.run(queue_item)              │
│       │           │                                         │
│       │           ├── run_node(invocation)                  │
│       │           │     │                                   │
│       │           │     ├── build_invocation_context        │
│       │           │     │                                   │
│       │           │     ├── invocation.invoke_internal()    │
│       │           │     │                                   │
│       │           │     └── session.complete(output)        │
│       │           │                                         │
│       │           └── on_after_run_session                  │
│       │                                                     │
│       └── Error handling                                    │
│                                                             │
│  Event Handlers                                             │
│       │                                                     │
│       ├── _on_queue_cleared() → cancel + poll_now           │
│       ├── _on_batch_enqueued() → poll_now                   │
│       └── _on_queue_item_status_changed() → cancel          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [Python Threading](https://docs.python.org/3/library/threading.html)
- [Python GC](https://docs.python.org/3/library/gc.html)
- [Producer-Consumer Pattern](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem)
