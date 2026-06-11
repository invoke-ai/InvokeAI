# توثيق ملف: boards.py

## مسار الملف الأصلي
```
invokeai/app/api/routers/boards.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/api/routers/boards.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **مسارات API للوحات** (Boards API Routes) في InvokeAI. يوفر endpoints لإدارة لوحات المستخدمين.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 مكتبات Python الأساسية
```python
from typing import Optional, Union
```

### 2.2 FastAPI
```python
from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter
```

### 2.3 Pydantic
```python
from pydantic import BaseModel, Field
```

### 2.4 مكتبات المشروع
```python
from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardRecordOrderBy, BoardVisibility
from invokeai.app.services.boards.boards_common import BoardDTO
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 تعريف الـ Router
```python
boards_router = APIRouter(prefix="/v1/boards", tags=["boards"])
```

### 3.2 نماذج البيانات

#### DeleteBoardResult
```python
class DeleteBoardResult(BaseModel):
    board_id: str = Field(description="The id of the board that was deleted.")
    deleted_board_images: list[str] = Field(description="The image names of the board-images relationships that were deleted.")
    deleted_images: list[str] = Field(description="The names of the images that were deleted.")
```

### 3.3 الـ Endpoints

#### إنشاء لوحة
```python
@boards_router.post(
    "/",
    operation_id="create_board",
    responses={201: {"description": "The board was created successfully"}},
    status_code=201,
    response_model=BoardDTO,
)
async def create_board(
    current_user: CurrentUserOrDefault,
    board_name: str = Query(description="The name of the board to create", max_length=300),
) -> BoardDTO:
    """Creates a board for the current user"""
    try:
        result = ApiDependencies.invoker.services.boards.create(board_name=board_name, user_id=current_user.user_id)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to create board")
```

#### الحصول على لوحة
```python
@boards_router.get("/{board_id}", operation_id="get_board", response_model=BoardDTO)
async def get_board(
    current_user: CurrentUserOrDefault,
    board_id: str = Path(description="The id of board to get"),
) -> BoardDTO:
    """Gets a board (user must have access to it)"""
    try:
        result = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Board not found")

    if (
        not current_user.is_admin
        and result.user_id != current_user.user_id
        and result.board_visibility == BoardVisibility.Private
    ):
        raise HTTPException(status_code=403, detail="Not authorized to access this board")

    return result
```

#### تحديث لوحة
```python
@boards_router.patch(
    "/{board_id}",
    operation_id="update_board",
    responses={201: {"description": "The board was updated successfully"}},
    status_code=201,
    response_model=BoardDTO,
)
async def update_board(
    current_user: CurrentUserOrDefault,
    board_id: str = Path(description="The id of board to update"),
    changes: BoardChanges = Body(description="The changes to apply to the board"),
) -> BoardDTO:
    """Updates a board (user must have access to it)"""
    try:
        board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Board not found")

    if not current_user.is_admin and board.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this board")

    try:
        result = ApiDependencies.invoker.services.boards.update(board_id=board_id, changes=changes)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to update board")
```

#### حذف لوحة
```python
@boards_router.delete("/{board_id}", operation_id="delete_board", response_model=DeleteBoardResult)
async def delete_board(
    current_user: CurrentUserOrDefault,
    board_id: str = Path(description="The id of board to delete"),
    include_images: Optional[bool] = Query(description="Permanently delete all images on the board", default=False),
) -> DeleteBoardResult:
    """Deletes a board (user must have access to it)"""
    try:
        board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Board not found")

    if not current_user.is_admin and board.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this board")

    try:
        if include_images is True:
            deleted_images = ApiDependencies.invoker.services.board_images.get_all_board_image_names_for_board(
                board_id=board_id, categories=None, is_intermediate=None,
            )
            ApiDependencies.invoker.services.images.delete_images_on_board(board_id=board_id)
            ApiDependencies.invoker.services.boards.delete(board_id=board_id)
            return DeleteBoardResult(board_id=board_id, deleted_board_images=[], deleted_images=deleted_images)
        else:
            deleted_board_images = ApiDependencies.invoker.services.board_images.get_all_board_image_names_for_board(
                board_id=board_id, categories=None, is_intermediate=None,
            )
            ApiDependencies.invoker.services.boards.delete(board_id=board_id)
            return DeleteBoardResult(board_id=board_id, deleted_board_images=deleted_board_images, deleted_images=[])
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete board")
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من الصلاحيات
```python
if (
    not current_user.is_admin
    and result.user_id != current_user.user_id
    and result.board_visibility == BoardVisibility.Private
):
    raise HTTPException(status_code=403, detail="Not authorized to access this board")
```

### 4.2 التعامل مع الأخطاء
```python
try:
    result = ApiDependencies.invoker.services.boards.create(board_name=board_name, user_id=current_user.user_id)
    return result
except Exception:
    raise HTTPException(status_code=500, detail="Failed to create board")
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **واجهة RESTful**: استخدام معايير REST.
2. **التحقق من الصلاحيات**: حماية البيانات.
3. **رسائل خطأ واضحة**: مساعدة المطورين.

### نقاط الضعف
1. **عدد كبير من الاستعلامات**: كل عملية تتطلب استعلامات متعددة.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Boards API Flow                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  /v1/boards                                                 │
│       │                                                     │
│       ├── POST /                                            │
│       │     └── create_board()                              │
│       │                                                     │
│       ├── GET /{board_id}                                   │
│       │     └── get_board()                                 │
│       │                                                     │
│       ├── PATCH /{board_id}                                 │
│       │     └── update_board()                              │
│       │                                                     │
│       ├── DELETE /{board_id}                                │
│       │     └── delete_board()                              │
│       │                                                     │
│       └── GET /                                             │
│             └── list_boards()                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [FastAPI Router](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [RESTful API Design](https://restfulapi.net/)
- [Authorization](https://en.wikipedia.org/wiki/Authorization)
