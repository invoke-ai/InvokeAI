# توثيق ملف: session_queue_sqlite.py

## مسار الملف الأصلي
```
invokeai/app/services/session_queue/session_queue_sqlite.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/services/session_queue/session_queue_sqlite.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **تنفيذ طابور الجلسات عبر SQLite** (SQLite Session Queue Implementation) الذي يوفر تخزيناً دائماً لعناصر الطابور في قاعدة بيانات SQLite.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 المكتبات المعيارية
```python
import asyncio
import json
import sqlite3
from typing import Optional, Union, cast
```

### 2.2 Pydantic Core
```python
from pydantic_core import to_jsonable_python
```

### 2.3 مكتبات المشروع
```python
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_base import SessionQueueBase
from invokeai.app.services.session_queue.session_queue_common import (
    DEFAULT_QUEUE_ID, QUEUE_ITEM_STATUS, Batch, BatchStatus,
    EnqueueBatchResult, SessionQueueItem, SessionQueueStatus, ...
)
from invokeai.app.services.shared.graph import GraphExecutionState
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 فئة SqliteSessionQueue

#### بدء التشغيل
```python
def start(self, invoker: Invoker) -> None:
    self.__invoker = invoker
    self._set_in_progress_to_canceled()
    config = self.__invoker.services.configuration

    if config.clear_queue_on_startup:
        clear_result = self.clear(DEFAULT_QUEUE_ID)
        if clear_result.deleted > 0:
            self.__invoker.services.logger.info(f"Cleared all {clear_result.deleted} queue items")
        return

    if config.max_queue_history is not None:
        deleted = self._prune_terminal_to_limit(DEFAULT_QUEUE_ID, config.max_queue_history)
        if deleted > 0:
            self.__invoker.services.logger.info(f"Pruned {deleted} completed/failed/canceled queue items")
```

#### تعيين العناصر الجارية كملغاة
```python
def _set_in_progress_to_canceled(self) -> None:
    with self._db.transaction() as cursor:
        cursor.execute("""
            UPDATE session_queue
            SET status = 'canceled',
                status_sequence = COALESCE(status_sequence, 0) + 1
            WHERE status = 'in_progress';
        """)
```

#### تقليم العناصر القديمة
```python
def _prune_terminal_to_limit(self, queue_id: str, keep: int) -> int:
    with self._db.transaction() as cursor:
        cursor.execute("""
            DELETE FROM session_queue
            WHERE queue_id = ?
            AND (status = 'completed' OR status = 'failed' OR status = 'canceled')
            AND item_id NOT IN (
                SELECT item_id FROM session_queue
                WHERE queue_id = ?
                AND (status = 'completed' OR status = 'failed' OR status = 'canceled')
                ORDER BY COALESCE(completed_at, updated_at, created_at) DESC, item_id DESC
                LIMIT ?
            );
        """, (queue_id, queue_id, keep))
    return count
```

#### الحصول على حجم الطابور
```python
def _get_current_queue_size(self, queue_id: str) -> int:
    with self._db.transaction() as cursor:
        cursor.execute("""
            SELECT count(*) FROM session_queue
            WHERE queue_id = ? AND status = 'pending';
        """, (queue_id,))
        count = cast(int, cursor.fetchone()[0])
    return count
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التعامل مع بدء التشغيل
- تعيين جميع العناصر الجارية كملغاة عند بدء التشغيل.
- تقليم العناصر القديمة حسب التكوين.

### 4.2 إدارة المعاملات
- استخدام `self._db.transaction()` لضمان نجاح العمليات.

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **تخزين دائم**: حفظ العناصر في SQLite.
2. **بدء تشغيل ذكي**: التعامل مع العناصر المتبقية من التشغيل السابق.
3. **تقليم تلقائي**: تقليم العناصر القديمة تلقائياً.

### نقاط الضعف
1. **أداء SQLite**: قد يكون أبطأ من التخزين في الذاكرة.
2. **تعقيد الاستعلامات**: بعض الاستعلامات معقدة.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              SQLite Queue Flow                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  start(invoker)                                             │
│       │                                                     │
│       ├── _set_in_progress_to_canceled()                    │
│       │     └── UPDATE status = 'canceled'                  │
│       │                                                     │
│       ├── clear_queue_on_startup?                           │
│       │     └── clear(DEFAULT_QUEUE_ID)                     │
│       │                                                     │
│       └── max_queue_history?                                │
│             └── _prune_terminal_to_limit()                  │
│                                                             │
│  enqueue_batch(queue_id, batch, prepend)                    │
│       │                                                     │
│       ├── _get_current_queue_size()                         │
│       ├── _get_highest_priority()                           │
│       ├── INSERT INTO session_queue                         │
│       └── Return EnqueueBatchResult                         │
│                                                             │
│  dequeue()                                                  │
│       │                                                     │
│       ├── SELECT ... WHERE status = 'pending'               │
│       ├── UPDATE status = 'in_progress'                     │
│       └── Return SessionQueueItem                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [SQLite Python](https://docs.python.org/3/library/sqlite3.html)
- [Queue Data Structure](https://en.wikipedia.org/wiki/Queue_(abstract_data_type))
- [Transaction Management](https://docs.python.org/3/library/sqlite3.html#transactions)
