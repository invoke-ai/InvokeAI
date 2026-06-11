# توثيق ملف: workflow_records_sqlite.py

## مسار الملف الأصلي
```
invokeai/app/services/workflow_records/workflow_records_sqlite.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/services/workflow_records/workflow_records_sqlite.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **تخزين سجلات سير العمل** (Workflow Records Storage) باستخدام SQLite. يدير إنشاء وتحديث وحذف سير العمل.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 مكتبات Python الأساسية
```python
from pathlib import Path
from typing import Optional
```

### 2.2 مكتبات المشروع
```python
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
from invokeai.app.services.workflow_records.workflow_records_common import (
    WORKFLOW_LIBRARY_DEFAULT_USER_ID, Workflow, WorkflowCategory, WorkflowNotFoundError,
    WorkflowRecordDTO, WorkflowRecordListItemDTO, WorkflowRecordListItemDTOValidator,
    WorkflowRecordOrderBy, WorkflowValidator, WorkflowWithoutID,
)
from invokeai.app.util.misc import uuid_string
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 ثوابت عامة
```python
SQL_TIME_FORMAT = "%Y-%m-%d %H:%M:%f"
```

### 3.2 فئة SqliteWorkflowRecordsStorage

#### التهيئة
```python
class SqliteWorkflowRecordsStorage(WorkflowRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_workflows()
```

#### الحصول على سير عمل
```python
def get(self, workflow_id: str) -> WorkflowRecordDTO:
    """Gets a workflow by ID. Updates the opened_at column."""
    with self._db.transaction() as cursor:
        cursor.execute(
            """--sql
            SELECT workflow_id, workflow, name, created_at, updated_at, opened_at, user_id, is_public
            FROM workflow_library
            WHERE workflow_id = ?;
            """,
            (workflow_id,),
        )
        row = cursor.fetchone()
    if row is None:
        raise WorkflowNotFoundError(f"Workflow with id {workflow_id} not found")
    return WorkflowRecordDTO.from_dict(dict(row))
```

#### إنشاء سير عمل
```python
def create(
    self,
    workflow: WorkflowWithoutID,
    user_id: str = WORKFLOW_LIBRARY_DEFAULT_USER_ID,
    is_public: bool = False,
) -> WorkflowRecordDTO:
    if workflow.meta.category is WorkflowCategory.Default:
        raise ValueError("Default workflows cannot be created via this method")

    with self._db.transaction() as cursor:
        workflow_with_id = Workflow(**workflow.model_dump(), id=uuid_string())
        cursor.execute(
            """--sql
            INSERT OR IGNORE INTO workflow_library (
                workflow_id, workflow, user_id, is_public
            ) VALUES (?, ?, ?, ?);
            """,
            (workflow_with_id.id, workflow_with_id.model_dump_json(), user_id, is_public),
        )
    return self.get(workflow_with_id.id)
```

#### تحديث سير عمل
```python
def update(self, workflow: Workflow, user_id: Optional[str] = None) -> WorkflowRecordDTO:
    if workflow.meta.category is WorkflowCategory.Default:
        raise ValueError("Default workflows cannot be updated")

    with self._db.transaction() as cursor:
        if user_id is not None:
            cursor.execute(
                """--sql
                UPDATE workflow_library
                SET workflow = ?
                WHERE workflow_id = ? AND category = 'user' AND user_id = ?;
                """,
                (workflow.model_dump_json(), workflow.id, user_id),
            )
        else:
            cursor.execute(
                """--sql
                UPDATE workflow_library
                SET workflow = ?
                WHERE workflow_id = ? AND category = 'user';
                """,
                (workflow.model_dump_json(), workflow.id),
            )
    return self.get(workflow.id)
```

#### حذف سير عمل
```python
def delete(self, workflow_id: str, user_id: Optional[str] = None) -> None:
    if self.get(workflow_id).workflow.meta.category is WorkflowCategory.Default:
        raise ValueError("Default workflows cannot be deleted")

    with self._db.transaction() as cursor:
        if user_id is not None:
            cursor.execute(
                """--sql
                DELETE from workflow_library
                WHERE workflow_id = ? AND category = 'user' AND user_id = ?;
                """,
                (workflow_id, user_id),
            )
        else:
            cursor.execute(
                """--sql
                DELETE from workflow_library
                WHERE workflow_id = ? AND category = 'user';
                """,
                (workflow_id,),
            )
    return None
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التعامل مع سير العمل غير الموجود
```python
if row is None:
    raise WorkflowNotFoundError(f"Workflow with id {workflow_id} not found")
```

### 4.2 التعامل مع سير العمل الافتراضي
```python
if workflow.meta.category is WorkflowCategory.Default:
    raise ValueError("Default workflows cannot be created via this method")
```

### 4.3 التعامل مع العلامة العامة
```python
def update_is_public(self, workflow_id: str, is_public: bool, user_id: Optional[str] = None) -> WorkflowRecordDTO:
    """Updates the is_public field of a workflow and manages the 'shared' tag automatically."""
    record = self.get(workflow_id)
    workflow = record.workflow

    tags_list = [t.strip() for t in workflow.tags.split(",") if t.strip()] if workflow.tags else []
    if is_public and "shared" not in tags_list:
        tags_list.append("shared")
    elif not is_public and "shared" in tags_list:
        tags_list.remove("shared")
    updated_tags = ", ".join(tags_list)
    updated_workflow = workflow.model_copy(update={"tags": updated_tags})
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **استخدام SQL**: أداء عالي لقاعدة البيانات.
2. **معالجة أخطاء واضحة**: رسائل خطأ مفيدة.
3. **flexibility**: دعم المستخدمين المتعددين.

### نقاط الضعف
1. **ال依赖 على SQLite**: قد لا يكون مناسباً للإنتاج.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Workflow Records SQLite Flow                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SqliteWorkflowRecordsStorage                               │
│       │                                                     │
│       ├── get(workflow_id)                                  │
│       │     ├── SELECT FROM workflow_library                │
│       │     └── Return WorkflowRecordDTO                    │
│       │                                                     │
│       ├── create(workflow, user_id, is_public)              │
│       │     ├── INSERT INTO workflow_library                │
│       │     └── Return new WorkflowRecordDTO                │
│       │                                                     │
│       ├── update(workflow, user_id)                         │
│       │     ├── UPDATE workflow_library                     │
│       │     └── Return updated WorkflowRecordDTO            │
│       │                                                     │
│       ├── delete(workflow_id, user_id)                      │
│       │     └── DELETE FROM workflow_library                │
│       │                                                     │
│       └── update_is_public(workflow_id, is_public)          │
│             ├── Manage "shared" tag                         │
│             └── UPDATE workflow_library                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [SQLite](https://www.sqlite.org/)
- [Workflow Management](https://en.wikipedia.org/wiki/Workflow_management_system)
- [CRUD Operations](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete)
