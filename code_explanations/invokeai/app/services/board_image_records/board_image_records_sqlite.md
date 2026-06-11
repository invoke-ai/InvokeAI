# توثيق ملف: board_image_records_sqlite.py

## مسار الملف الأصلي
```
invokeai/app/services/board_image_records/board_image_records_sqlite.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/services/board_image_records/board_image_records_sqlite.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **تخزين سجلات صور اللوحات** (Board Image Records Storage) باستخدام SQLite. يدير العلاقة بين اللوحات والصور.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 مكتبات Python الأساسية
```python
import sqlite3
from typing import Optional, cast
```

### 2.2 مكتبات المشروع
```python
from invokeai.app.services.board_image_records.board_image_records_base import BoardImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import (
    ASSETS_CATEGORIES, IMAGE_CATEGORIES, ImageCategory, ImageRecord, deserialize_image_record,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 فئة SqliteBoardImageRecordStorage

#### التهيئة
```python
class SqliteBoardImageRecordStorage(BoardImageRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
```

#### إضافة صورة إلى لوحة
```python
def add_image_to_board(self, board_id: str, image_name: str) -> None:
    with self._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO board_images (board_id, image_name)
            VALUES (?, ?)
            ON CONFLICT (image_name) DO UPDATE SET board_id = ?;
            """,
            (board_id, image_name, board_id),
        )
```

#### إزالة صورة من لوحة
```python
def remove_image_from_board(self, image_name: str) -> None:
    with self._db.transaction() as cursor:
        cursor.execute(
            """--sql
            DELETE FROM board_images
            WHERE image_name = ?;
            """,
            (image_name,),
        )
```

#### الحصول على صور اللوحة
```python
def get_images_for_board(
    self,
    board_id: str,
    offset: int = 0,
    limit: int = 10,
) -> OffsetPaginatedResults[ImageRecord]:
    with self._db.transaction() as cursor:
        cursor.execute(
            """--sql
            SELECT images.*
            FROM board_images
            INNER JOIN images ON board_images.image_name = images.image_name
            WHERE board_images.board_id = ?
            ORDER BY board_images.updated_at DESC;
            """,
            (board_id,),
        )
        result = cast(list[sqlite3.Row], cursor.fetchall())
        images = [deserialize_image_record(dict(r)) for r in result]

        cursor.execute(
            """--sql
            SELECT COUNT(*) FROM images WHERE 1=1;
            """
        )
        count = cast(int, cursor.fetchone()[0])

    return OffsetPaginatedResults(items=images, offset=offset, limit=limit, total=count)
```

#### الحصول على أسماء الصور للوحة
```python
def get_all_board_image_names_for_board(
    self,
    board_id: str,
    categories: list[ImageCategory] | None,
    is_intermediate: bool | None,
) -> list[str]:
    with self._db.transaction() as cursor:
        params: list[str | bool] = []

        stmt = """
                SELECT images.image_name
                FROM images
                LEFT JOIN board_images ON board_images.image_name = images.image_name
                WHERE 1=1
                """

        if board_id == "none":
            stmt += """--sql
                AND board_images.board_id IS NULL
                """
        else:
            stmt += """--sql
                AND board_images.board_id = ?
                """
            params.append(board_id)

        if categories is not None:
            category_strings = [c.value for c in set(categories)]
            placeholders = ",".join("?" * len(category_strings))
            stmt += f"""--sql
                AND images.image_category IN ( {placeholders} )
                """
            for c in category_strings:
                params.append(c)

        if is_intermediate is not None:
            stmt += """--sql
                AND images.is_intermediate = ?
                """
            params.append(is_intermediate)

        stmt += ";"
        cursor.execute(stmt, params)

        result = cast(list[sqlite3.Row], cursor.fetchall())
    image_names = [r[0] for r in result]
    return image_names
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التعامل مع التعارض
```python
INSERT INTO board_images (board_id, image_name)
VALUES (?, ?)
ON CONFLICT (image_name) DO UPDATE SET board_id = ?;
```

### 4.2 التعامل مع اللوحات الفارغة
```python
if board_id == "none":
    stmt += """--sql
        AND board_images.board_id IS NULL
        """
```

### 4.3 التعامل مع التصنيفات
```python
if categories is not None:
    category_strings = [c.value for c in set(categories)]
    placeholders = ",".join("?" * len(category_strings))
    stmt += f"""--sql
        AND images.image_category IN ( {placeholders} )
        """
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **استخدام SQL**: أداء عالي لقاعدة البيانات.
2. **معالجة أخطاء واضحة**: رسائل خطأ مفيدة.
3. **flexibility**: دعم التصنيفات المختلفة.

### نقاط الضعف
1. **ال依赖 على SQLite**: قد لا يكون مناسباً للإنتاج.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Board Image Records Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SqliteBoardImageRecordStorage                              │
│       │                                                     │
│       ├── add_image_to_board(board_id, image_name)          │
│       │     ├── INSERT INTO board_images                    │
│       │     └── ON CONFLICT UPDATE                          │
│       │                                                     │
│       ├── remove_image_from_board(image_name)               │
│       │     └── DELETE FROM board_images                    │
│       │                                                     │
│       ├── get_images_for_board(board_id)                    │
│       │     ├── SELECT FROM board_images JOIN images        │
│       │     └── Return OffsetPaginatedResults               │
│       │                                                     │
│       └── get_all_board_image_names_for_board(board_id)     │
│             ├── Build dynamic SQL query                     │
│             ├── Apply filters                               │
│             └── Return list of image names                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [SQLite](https://www.sqlite.org/)
- [SQL JOIN](https://www.w3schools.com/sql/sql_join.asp)
- [CRUD Operations](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete)
