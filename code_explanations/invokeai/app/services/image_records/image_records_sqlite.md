# توثيق ملف: image_records_sqlite.py

## مسار الملف الأصلي
```
invokeai/app/services/image_records/image_records_sqlite.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/services/image_records/image_records_sqlite.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **تخزين سجلات الصور** (Image Records Storage) باستخدام SQLite. يدير إنشاء وتحديث وحذف سجلات الصور.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 مكتبات Python الأساسية
```python
import sqlite3
from datetime import datetime
from typing import Optional, Union, cast
```

### 2.2 مكتبات المشروع
```python
from invokeai.app.invocations.fields import MetadataField, MetadataFieldValidator
from invokeai.app.services.image_records.image_records_base import ImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import (
    IMAGE_DTO_COLS, ImageCategory, ImageNamesResult, ImageRecord, ImageRecordChanges,
    ImageRecordDeleteException, ImageRecordNotFoundException, ImageRecordSaveException,
    ResourceOrigin, deserialize_image_record,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.virtual_boards.virtual_boards_common import VirtualSubBoardDTO
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 فئة SqliteImageRecordStorage

#### التهيئة
```python
class SqliteImageRecordStorage(ImageRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
```

#### الحصول على صورة
```python
def get(self, image_name: str) -> ImageRecord:
    with self._db.transaction() as cursor:
        try:
            cursor.execute(
                f"""--sql
                SELECT {IMAGE_DTO_COLS} FROM images
                WHERE image_name = ?;
                """,
                (image_name,),
            )
            result = cast(Optional[sqlite3.Row], cursor.fetchone())
        except sqlite3.Error as e:
            raise ImageRecordNotFoundException from e

    if not result:
        raise ImageRecordNotFoundException

    return deserialize_image_record(dict(result))
```

#### الحصول على معرف المستخدم
```python
def get_user_id(self, image_name: str) -> Optional[str]:
    with self._db.transaction() as cursor:
        cursor.execute(
            """--sql
            SELECT user_id FROM images
            WHERE image_name = ?;
            """,
            (image_name,),
        )
        result = cast(Optional[sqlite3.Row], cursor.fetchone())
        if not result:
            return None
        return cast(Optional[str], dict(result).get("user_id"))
```

#### الحصول على البيانات الوصفية
```python
def get_metadata(self, image_name: str) -> Optional[MetadataField]:
    with self._db.transaction() as cursor:
        try:
            cursor.execute(
                """--sql
                SELECT metadata FROM images
                WHERE image_name = ?;
                """,
                (image_name,),
            )
            result = cast(Optional[sqlite3.Row], cursor.fetchone())
        except sqlite3.Error as e:
            raise ImageRecordNotFoundException from e

        if not result:
            raise ImageRecordNotFoundException

        as_dict = dict(result)
        metadata_raw = cast(Optional[str], as_dict.get("metadata", None))
        return MetadataFieldValidator.validate_json(metadata_raw) if metadata_raw is not None else None
```

#### تحديث صورة
```python
def update(self, image_name: str, changes: ImageRecordChanges) -> None:
    with self._db.transaction() as cursor:
        try:
            if changes.image_category is not None:
                cursor.execute(
                    """--sql
                    UPDATE images SET image_category = ? WHERE image_name = ?;
                    """,
                    (changes.image_category, image_name),
                )

            if changes.session_id is not None:
                cursor.execute(
                    """--sql
                    UPDATE images SET session_id = ? WHERE image_name = ?;
                    """,
                    (changes.session_id, image_name),
                )

            if changes.is_intermediate is not None:
                cursor.execute(
                    """--sql
                    UPDATE images SET is_intermediate = ? WHERE image_name = ?;
                    """,
                    (changes.is_intermediate, image_name),
                )

            if changes.starred is not None:
                cursor.execute(
                    """--sql
                    UPDATE images SET starred = ? WHERE image_name = ?;
                    """,
                    (changes.starred, image_name),
                )

        except sqlite3.Error as e:
            raise ImageRecordSaveException from e
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التعامل مع الصور غير الموجودة
```python
if not result:
    raise ImageRecordNotFoundException
```

### 4.2 التعامل مع أخطاء SQLite
```python
try:
    cursor.execute(...)
except sqlite3.Error as e:
    raise ImageRecordNotFoundException from e
```

### 4.3 التعامل مع البيانات الوصفية
```python
metadata_raw = cast(Optional[str], as_dict.get("metadata", None))
return MetadataFieldValidator.validate_json(metadata_raw) if metadata_raw is not None else None
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **استخدام SQL**: أداء عالي لقاعدة البيانات.
2. **معالجة أخطاء واضحة**: رسائل خطأ مفيدة.
3. **flexibility**: دعم أنواع مختلفة من التحديثات.

### نقاط الضعف
1. **ال依赖 على SQLite**: قد لا يكون مناسباً للإنتاج.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Image Records SQLite Flow                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SqliteImageRecordStorage                                   │
│       │                                                     │
│       ├── get(image_name)                                   │
│       │     ├── SELECT FROM images                          │
│       │     └── Return ImageRecord                          │
│       │                                                     │
│       ├── get_user_id(image_name)                           │
│       │     ├── SELECT user_id FROM images                  │
│       │     └── Return Optional[str]                        │
│       │                                                     │
│       ├── get_metadata(image_name)                          │
│       │     ├── SELECT metadata FROM images                 │
│       │     └── Return Optional[MetadataField]              │
│       │                                                     │
│       └── update(image_name, changes)                       │
│             ├── UPDATE images SET image_category            │
│             ├── UPDATE images SET session_id                │
│             ├── UPDATE images SET is_intermediate           │
│             └── UPDATE images SET starred                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [SQLite](https://www.sqlite.org/)
- [Image Metadata](https://en.wikipedia.org/wiki/Exchangeable_image_file_format)
- [CRUD Operations](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete)
