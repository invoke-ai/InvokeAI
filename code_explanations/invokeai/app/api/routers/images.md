# توثيق ملف: images.py

## مسار الملف الأصلي
```
invokeai/app/api/routers/images.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/api/routers/images.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **محور الصور** (Images Router) الذي يوفر واجهة برمجة تطبيقات REST لإدارة الصور. وهو مسؤول عن الرفع، والحذف، والتحديث، والبحث، وتنزيل الصور.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 FastAPI
```python
from fastapi import BackgroundTasks, Body, HTTPException, Path, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
```

### 2.2 PIL (Pillow)
```python
from PIL import Image
```

### 2.3 Pydantic
```python
from pydantic import BaseModel, Field, model_validator
```

### 2.4 مكتبات المشروع
```python
from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageNamesResult, ImageRecordChanges
from invokeai.app.services.images.images_common import DeleteImagesResult, ImageDTO, ImageUrlsDTO
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 تعريف المحور
```python
images_router = APIRouter(prefix="/v1/images", tags=["images"])
IMAGE_MAX_AGE = 31536000
```

### 3.2 نموذج تغيير الأبعاد
```python
class ResizeToDimensions(BaseModel):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    MAX_SIZE: ClassVar[int] = 4096 * 4096

    @model_validator(mode="after")
    def validate_total_output_size(self):
        if self.width * self.height > self.MAX_SIZE:
            raise ValueError(f"Max total output size for resizing is {self.MAX_SIZE} pixels")
        return self
```

### 3.3 نقاط النهاية (Endpoints)

#### upload_image
```python
@images_router.post("/upload", operation_id="upload_image", status_code=201, response_model=ImageDTO)
async def upload_image(
    current_user: CurrentUserOrDefault,
    file: UploadFile,
    request: Request,
    response: Response,
    image_category: ImageCategory = Query(description="The category of the image"),
    is_intermediate: bool = Query(description="Whether this is an intermediate image"),
    board_id: Optional[str] = Query(default=None, description="The board to add this image to"),
    session_id: Optional[str] = Query(default=None, description="The session ID associated with this upload"),
    crop_visible: Optional[bool] = Query(default=False, description="Whether to crop the image"),
    resize_to: Optional[str] = Body(default=None, description="Dimensions to resize the image to"),
    metadata: Optional[str] = Body(default=None, description="The metadata to associate with the image"),
) -> ImageDTO:
    # التحقق من الصلاحيات
    if board_id is not None:
        board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
        if not current_user.is_admin and board.user_id != current_user.user_id and board.board_visibility != BoardVisibility.Public:
            raise HTTPException(status_code=403, detail="Not authorized to upload to this board")

    # التحقق من نوع الملف
    if not file.content_type or not file.content_type.startswith("image"):
        raise HTTPException(status_code=415, detail="Not an image")

    # قراءة الصورة
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    # القص
    if crop_visible:
        bbox = pil_image.getbbox()
        pil_image = pil_image.crop(bbox)

    # تغيير الحجم
    if resize_to:
        dims = json.loads(resize_to)
        resize_dims = ResizeToDimensions(**dims)
        pil_rgba = pil_image.convert("RGBA")
        np_image = pil_to_np(pil_rgba)
        np_image = heuristic_resize_fast(np_image, (resize_dims.width, resize_dims.height))
        pil_image = np_to_pil(np_image)

    # استخراج البيانات الوصفية
    extracted_metadata = extract_metadata_from_image(pil_image=pil_image, ...)

    # إنشاء الصورة
    image_dto = ApiDependencies.invoker.services.images.create(
        image=pil_image, image_origin=ResourceOrigin.EXTERNAL, image_category=image_category, ...
    )

    response.status_code = 201
    response.headers["Location"] = image_dto.image_url
    return image_dto
```

#### delete_image
```python
@images_router.delete("/i/{image_name}", operation_id="delete_image", response_model=DeleteImagesResult)
async def delete_image(
    current_user: CurrentUserOrDefault,
    image_name: str = Path(description="The name of the image to delete"),
) -> DeleteImagesResult:
    _assert_image_owner(image_name, current_user)
    ApiDependencies.invoker.services.images.delete(image_name)
    return DeleteImagesResult(deleted_images=list(deleted_images), affected_boards=list(affected_boards))
```

#### get_image_dto
```python
@images_router.get("/i/{image_name}", operation_id="get_image_dto", response_model=ImageDTO)
async def get_image_dto(
    current_user: CurrentUserOrDefault,
    image_name: str = Path(description="The name of image to get"),
) -> ImageDTO:
    _assert_image_read_access(image_name, current_user)
    return ApiDependencies.invoker.services.images.get_dto(image_name)
```

#### get_image_full
```python
@images_router.get("/i/{image_name}/full", operation_id="get_image_full", response_class=Response)
async def get_image_full(image_name: str = Path(description="The name of full-resolution image file to get")) -> Response:
    path = ApiDependencies.invoker.services.images.get_path(image_name)
    with open(path, "rb") as f:
        content = f.read()
    response = Response(content, media_type="image/png")
    response.headers["Cache-Control"] = f"max-age={IMAGE_MAX_AGE}"
    return response
```

#### list_image_dtos
```python
@images_router.get("/", operation_id="list_image_dtos", response_model=OffsetPaginatedResults[ImageDTO])
async def list_image_dtos(
    current_user: CurrentUserOrDefault,
    image_origin: Optional[ResourceOrigin] = Query(default=None, description="The origin of images to list."),
    categories: Optional[list[ImageCategory]] = Query(default=None, description="The categories of image to include."),
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate images."),
    board_id: Optional[str] = Query(default=None, description="The board id to filter by."),
    offset: int = Query(default=0, description="The page offset"),
    limit: int = Query(default=10, description="The number of images per page"),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred images first"),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
) -> OffsetPaginatedResults[ImageDTO]:
    if board_id is not None and board_id != "none":
        _assert_board_read_access(board_id, current_user)
    image_dtos = ApiDependencies.invoker.services.images.get_many(
        offset, limit, starred_first, order_dir, image_origin, categories, is_intermediate, board_id, search_term, current_user.user_id
    )
    return image_dtos
```

#### download_images_from_list
```python
@images_router.post("/download", operation_id="download_images_from_list", response_model=ImagesDownloaded, status_code=202)
async def download_images_from_list(
    current_user: CurrentUserOrDefault,
    background_tasks: BackgroundTasks,
    image_names: Optional[list[str]] = Body(default=None, description="The list of names of images to download"),
    board_id: Optional[str] = Body(default=None, description="The board from which image should be downloaded"),
) -> ImagesDownloaded:
    if (image_names is None or len(image_names) == 0) and board_id is None:
        raise HTTPException(status_code=400, detail="No images or board id specified.")
    bulk_download_item_id = ApiDependencies.invoker.services.bulk_download.generate_item_id(board_id)
    background_tasks.add_task(ApiDependencies.invoker.services.bulk_download.handler, image_names, board_id, bulk_download_item_id, current_user.user_id)
    return ImagesDownloaded(bulk_download_item_name=bulk_download_item_id + ".zip")
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من الصلاحيات
```python
_assert_image_owner(image_name, current_user)
_assert_image_read_access(image_name, current_user)
_assert_board_read_access(board_id, current_user)
```

### 4.2 التحقق من نوع الملف
```python
if not file.content_type or not file.content_type.startswith("image"):
    raise HTTPException(status_code=415, detail="Not an image")
```

### 4.3 التحقق من الأبعاد
```python
if self.width * self.height > self.MAX_SIZE:
    raise ValueError(f"Max total output size for resizing is {self.MAX_SIZE} pixels")
```

### 4.4 التعامل مع عدم وجود الصور
```python
try:
    return ApiDependencies.invoker.services.images.get_dto(image_name)
except Exception:
    raise HTTPException(status_code=404)
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **صلاحيات متعددة المستخدمين**: دعم كامل لمستخدمين متعددين.
2. **تنظيف البيانات**: حماية بيانات المستخدمين الآخرين.
3. **دعم التحميل الخلفي**: استخدام BackgroundTasks للتحميل.
4. **توثيق شامل**: كل نقطة نهاية موثقة بوضوح.

### نقاط الضعف
1. **عدد كبير من نقاط النهاية**: قد يصعب الصيانة.
2. **تكرار الكود**: بعض الأكواد مكررة.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Images Router API Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  POST /upload                                               │
│       ├── Validate board access                             │
│       ├── Read file                                         │
│       ├── Crop if needed                                    │
│       ├── Resize if needed                                  │
│       ├── Extract metadata                                  │
│       └── Create image                                      │
│                                                             │
│  DELETE /i/{image_name}                                     │
│       └── Delete image (owner only)                         │
│                                                             │
│  GET /i/{image_name}                                        │
│       └── Get image DTO                                     │
│                                                             │
│  GET /i/{image_name}/full                                   │
│       └── Get full-resolution image                         │
│                                                             │
│  GET /i/{image_name}/thumbnail                              │
│       └── Get thumbnail                                     │
│                                                             │
│  GET /                                                      │
│       └── List images with pagination                       │
│                                                             │
│  POST /download                                             │
│       └── Download images (background task)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [FastAPI File Upload](https://fastapi.tiangolo.com/tutorial/request-files/)
- [PIL Image Processing](https://pillow.readthedocs.io/)
- [REST API Pagination](https://restfulapi.net/paging/)
