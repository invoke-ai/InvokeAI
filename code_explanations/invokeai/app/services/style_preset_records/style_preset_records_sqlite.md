# توثيق ملف: style_preset_records_sqlite.py

## مسار الملف الأصلي
```
invokeai/app/services/style_preset_records/style_preset_records_sqlite.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/services/style_preset_records/style_preset_records_sqlite.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **تخزين سجلات الإعدادات النمطية** (Style Preset Records Storage) باستخدام SQLite. يدير إنشاء وتحديث وحذف الإعدادات النمطية.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 مكتبات Python الأساسية
```python
import json
from pathlib import Path
```

### 2.2 مكتبات المشروع
```python
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.style_preset_records.style_preset_records_base import StylePresetRecordsStorageBase
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetType, StylePresetChanges, StylePresetNotFoundError, StylePresetRecordDTO, StylePresetWithoutId,
)
from invokeai.app.util.misc import uuid_string
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 ثوابت عامة
```python
SYSTEM_USER_ID = "system"
```

### 3.2 فئة SqliteStylePresetRecordsStorage

#### التهيئة
```python
class SqliteStylePresetRecordsStorage(StylePresetRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_style_presets()
```

#### الحصول على إعداد نمطي
```python
def get(self, style_preset_id: str) -> StylePresetRecordDTO:
    """Gets a style preset by ID."""
    with self._db.transaction() as cursor:
        cursor.execute(
            """--sql
            SELECT *
            FROM style_presets
            WHERE id = ?;
            """,
            (style_preset_id,),
        )
        row = cursor.fetchone()
    if row is None:
        raise StylePresetNotFoundError(f"Style preset with id {style_preset_id} not found")
    return StylePresetRecordDTO.from_dict(dict(row))
```

#### إنشاء إعداد نمطي
```python
def create(self, style_preset: StylePresetWithoutId, user_id: str) -> StylePresetRecordDTO:
    style_preset_id = uuid_string()
    with self._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT OR IGNORE INTO style_presets (
                id, name, preset_data, type, user_id, is_public
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                style_preset_id,
                style_preset.name,
                style_preset.preset_data.model_dump_json(),
                style_preset.type,
                user_id,
                1 if style_preset.is_public else 0,
            ),
        )
    return self.get(style_preset_id)
```

#### إنشاء إعدادات متعددة
```python
def create_many(self, style_presets: list[StylePresetWithoutId], user_id: str) -> None:
    with self._db.transaction() as cursor:
        for style_preset in style_presets:
            style_preset_id = uuid_string()
            cursor.execute(
                """--sql
                INSERT OR IGNORE INTO style_presets (
                    id, name, preset_data, type, user_id, is_public
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    style_preset_id,
                    style_preset.name,
                    style_preset.preset_data.model_dump_json(),
                    style_preset.type,
                    user_id,
                    1 if style_preset.is_public else 0,
                ),
            )
    return None
```

#### تحديث إعداد نمطي
```python
def update(self, style_preset_id: str, changes: StylePresetChanges) -> StylePresetRecordDTO:
    with self._db.transaction() as cursor:
        if changes.name is not None:
            cursor.execute(
                """--sql
                UPDATE style_presets SET name = ? WHERE id = ?;
                """,
                (changes.name, style_preset_id),
            )

        if changes.preset_data is not None:
            cursor.execute(
                """--sql
                UPDATE style_presets SET preset_data = ? WHERE id = ?;
                """,
                (changes.preset_data.model_dump_json(), style_preset_id),
            )

        if changes.is_public is not None:
            cursor.execute(
                """--sql
                UPDATE style_presets SET is_public = ? WHERE id = ?;
                """,
                (1 if changes.is_public else 0, style_preset_id),
            )

    return self.get(style_preset_id)
```

#### حذف إعداد نمطي
```python
def delete(self, style_preset_id: str) -> None:
    with self._db.transaction() as cursor:
        cursor.execute(
            """--sql
            DELETE from style_presets WHERE id = ?;
            """,
            (style_preset_id,),
        )
    return None
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التعامل مع الإعدادات غير الموجودة
```python
if row is None:
    raise StylePresetNotFoundError(f"Style preset with id {style_preset_id} not found")
```

### 4.2 التعامل مع التعارض
```python
INSERT OR IGNORE INTO style_presets (
    id, name, preset_data, type, user_id, is_public
) VALUES (?, ?, ?, ?, ?, ?);
```

### 4.3 التعامل مع البيانات الوصفية
```python
style_preset.preset_data.model_dump_json()
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **استخدام SQL**: أداء عالي لقاعدة البيانات.
2. **معالجة أخطاء واضحة**: رسائل خطأ مفيدة.
3. **flexibility**: دعم أنواع مختلفة من الإعدادات.

### نقاط الضعف
1. **ال依赖 على SQLite**: قد لا يكون مناسباً للإنتاج.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Style Preset Records Flow                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SqliteStylePresetRecordsStorage                            │
│       │                                                     │
│       ├── get(style_preset_id)                              │
│       │     ├── SELECT FROM style_presets                   │
│       │     └── Return StylePresetRecordDTO                 │
│       │                                                     │
│       ├── create(style_preset, user_id)                     │
│       │     ├── INSERT INTO style_presets                   │
│       │     └── Return new StylePresetRecordDTO             │
│       │                                                     │
│       ├── create_many(style_presets, user_id)               │
│       │     └── INSERT for each style preset                │
│       │                                                     │
│       ├── update(style_preset_id, changes)                  │
│       │     ├── UPDATE style_presets                        │
│       │     └── Return updated StylePresetRecordDTO         │
│       │                                                     │
│       └── delete(style_preset_id)                           │
│             └── DELETE FROM style_presets                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [SQLite](https://www.sqlite.org/)
- [Style Presets](https://en.wikipedia.org/wiki/Style_transfer)
- [CRUD Operations](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete)
