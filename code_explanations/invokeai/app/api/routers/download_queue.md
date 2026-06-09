# توثيق ملف: download_queue.py

## مسار الملف الأصلي
```
invokeai/app/api/routers/download_queue.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/api/routers/download_queue.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **مسارات API لطابور التنزيل** (Download Queue API Routes) في InvokeAI. يوفر endpoints لإدارة عمليات التنزيل.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 مكتبات Python الأساسية
```python
from pathlib import Path as FsPath
from pathlib import PurePosixPath, PureWindowsPath
from typing import List, Optional
```

### 2.2 FastAPI
```python
from fastapi import Body, Path, Response
from fastapi.routing import APIRouter
```

### 2.3 Pydantic
```python
from pydantic.networks import AnyHttpUrl
```

### 2.4 Starlette
```python
from starlette.exceptions import HTTPException
```

### 2.5 مكتبات المشروع
```python
from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.download import DownloadJob, UnknownJobIDException
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 تعريف الـ Router
```python
download_queue_router = APIRouter(prefix="/v1/download_queue", tags=["download_queue"])
```

### 3.2 الدالة المساعدة للتحقق من المسار
```python
def _validate_dest(dest: str) -> str:
    """Reject absolute paths and parent-traversal segments."""
    if not dest or not dest.strip():
        raise HTTPException(status_code=400, detail="Download destination must not be empty.")

    posix = PurePosixPath(dest)
    windows = PureWindowsPath(dest)
    if posix.is_absolute() or windows.is_absolute():
        raise HTTPException(status_code=400, detail="Download destination must be a relative path.")

    if ".." in posix.parts or ".." in windows.parts:
        raise HTTPException(status_code=400, detail="Download destination must not contain '..' segments.")

    return dest
```

### 3.3 الـ Endpoints

#### قائمة التنزيلات
```python
@download_queue_router.get(
    "/",
    operation_id="list_downloads",
)
async def list_downloads(current_user: CurrentUserOrDefault) -> List[DownloadJob]:
    """Get a list of active and inactive jobs."""
    queue = ApiDependencies.invoker.services.download_queue
    return queue.list_jobs()
```

#### تنظيف التنزيلات
```python
@download_queue_router.patch(
    "/",
    operation_id="prune_downloads",
    responses={204: {"description": "All completed jobs have been pruned"}, 400: {"description": "Bad request"}},
)
async def prune_downloads(current_user: AdminUserOrDefault) -> Response:
    """Prune completed and errored jobs."""
    queue = ApiDependencies.invoker.services.download_queue
    queue.prune_jobs()
    return Response(status_code=204)
```

#### تنزيل ملف
```python
@download_queue_router.post(
    "/i/",
    operation_id="download",
)
async def download(
    current_user: CurrentUserOrDefault,
    source: AnyHttpUrl = Body(description="download source"),
    dest: str = Body(description="download destination"),
    priority: int = Body(default=10, description="queue priority"),
    access_token: Optional[str] = Body(default=None, description="token for authorization to download"),
) -> DownloadJob:
    """Download the source URL to the file or directory indicted in dest."""
    validated_dest = _validate_dest(dest)
    queue = ApiDependencies.invoker.services.download_queue
    return queue.download(source, FsPath(validated_dest), priority, access_token)
```

#### الحصول على عمل تنزيل
```python
@download_queue_router.get(
    "/i/{id}",
    operation_id="get_download_job",
    responses={200: {"description": "Success"}, 404: {"description": "The requested download JobID could not be found"}},
)
async def get_download_job(
    current_user: CurrentUserOrDefault,
    id: int = Path(description="ID of the download job to fetch."),
) -> DownloadJob:
    """Get a download job using its ID."""
    try:
        job = ApiDependencies.invoker.services.download_queue.id_to_job(id)
        return job
    except UnknownJobIDException as e:
        raise HTTPException(status_code=404, detail=str(e))
```

#### إلغاء عمل تنزيل
```python
@download_queue_router.delete(
    "/i/{id}",
    operation_id="cancel_download_job",
    responses={204: {"description": "Job has been cancelled"}, 404: {"description": "The requested download JobID could not be found"}},
)
async def cancel_download_job(
    current_user: CurrentUserOrDefault,
    id: int = Path(description="ID of the download job to cancel."),
) -> Response:
    """Cancel a download job using its ID."""
    try:
        queue = ApiDependencies.invoker.services.download_queue
        job = queue.id_to_job(id)
        queue.cancel_job(job)
        return Response(status_code=204)
    except UnknownJobIDException as e:
        raise HTTPException(status_code=404, detail=str(e))
```

#### إلغاء جميع التنزيلات
```python
@download_queue_router.delete(
    "/i",
    operation_id="cancel_all_download_jobs",
    responses={204: {"description": "Download jobs have been cancelled"}},
)
async def cancel_all_download_jobs(current_user: AdminUserOrDefault) -> Response:
    """Cancel all download jobs."""
    ApiDependencies.invoker.services.download_queue.cancel_all_jobs()
    return Response(status_code=204)
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من صحة المسار
```python
def _validate_dest(dest: str) -> str:
    """Reject absolute paths and parent-traversal segments."""
    if not dest or not dest.strip():
        raise HTTPException(status_code=400, detail="Download destination must not be empty.")

    posix = PurePosixPath(dest)
    windows = PureWindowsPath(dest)
    if posix.is_absolute() or windows.is_absolute():
        raise HTTPException(status_code=400, detail="Download destination must be a relative path.")

    if ".." in posix.parts or ".." in windows.parts:
        raise HTTPException(status_code=400, detail="Download destination must not contain '..' segments.")

    return dest
```

### 4.2 التعامل مع أخطاء التنزيل
```python
try:
    job = ApiDependencies.invoker.services.download_queue.id_to_job(id)
    return job
except UnknownJobIDException as e:
    raise HTTPException(status_code=404, detail=str(e))
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **واجهة RESTful**: استخدام معايير REST.
2. **التحقق من المدخلات**: حماية النظام من المدخلات غير الصالحة.
3. **رسائل خطأ واضحة**: مساعدة المطورين.

### نقاط الضعف
1. **أمان محدود**: لا يوجد تحقق من الصلاحيات للعمليات.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Download Queue API Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  /v1/download_queue                                         │
│       │                                                     │
│       ├── GET /                                             │
│       │     └── list_downloads()                            │
│       │                                                     │
│       ├── PATCH /                                           │
│       │     └── prune_downloads()                           │
│       │                                                     │
│       ├── POST /i/                                          │
│       │     └── download()                                  │
│       │                                                     │
│       ├── GET /i/{id}                                       │
│       │     └── get_download_job()                          │
│       │                                                     │
│       ├── DELETE /i/{id}                                    │
│       │     └── cancel_download_job()                       │
│       │                                                     │
│       └── DELETE /i                                         │
│             └── cancel_all_download_jobs()                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [FastAPI Router](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [RESTful API Design](https://restfulapi.net/)
- [Download Management](https://en.wikipedia.org/wiki/Download_manager)
