# توثيق ملف: invocation_cache_memory.py

## مسار الملف الأصلي
```
invokeai/app/services/invocation_cache/invocation_cache_memory.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/services/invocation_cache/invocation_cache_memory.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **نظام التخزين المؤقت في الذاكرة** (In-Memory Cache) لنتائج العقد. وهو يُستخدم لتخزين نتائج العقد المُنجزة لتجنب إعادة حسابها، مما يحسّن أداء التطبيق بشكل كبير.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 المكتبات المعيارية
```python
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional, Union
```
- **OrderedDict**: قاموس مرتب يحافظ على ترتيب الإدراج.
- **dataclass**: لإنشاء فئات بيانات بسيطة.
- **Lock**: للتعامل مع الخيوط المتعددة.

### 2.2 مكتبات المشروع
```python
from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase
from invokeai.app.services.invocation_cache.invocation_cache_common import InvocationCacheStatus
from invokeai.app.services.invoker import Invoker
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 فئة CachedItem
```python
@dataclass(order=True)
class CachedItem:
    invocation_output: BaseInvocationOutput = field(compare=False)
    invocation_output_json: str = field(compare=False)
```
- **الهدف**: تخزين مخرجات العقد وتمثيلها كنص JSON.

### 3.2 فئة MemoryInvocationCache

#### الخصائص
```python
class MemoryInvocationCache(InvocationCacheBase):
    _cache: OrderedDict[Union[int, str], CachedItem]
    _max_cache_size: int
    _disabled: bool
    _hits: int
    _misses: int
    _invoker: Invoker
    _lock: Lock
```

#### التهيئة
```python
def __init__(self, max_cache_size: int = 0) -> None:
    self._cache = OrderedDict()
    self._max_cache_size = max_cache_size
    self._disabled = False
    self._hits = 0
    self._misses = 0
    self._lock = Lock()
```

#### بدء التشغيل
```python
def start(self, invoker: Invoker) -> None:
    self._invoker = invoker
    if self._max_cache_size == 0:
        return
    self._invoker.services.images.on_deleted(self._delete_by_match)
    self._invoker.services.tensors.on_deleted(self._delete_by_match)
    self._invoker.services.conditioning.on_deleted(self._delete_by_match)
```
- **الهدف**: تسجيل مستمعين لحذف العناصر المؤقتة عند حذف الصور أو المصفوفات.

#### جلب النتائج
```python
def get(self, key: Union[int, str]) -> Optional[BaseInvocationOutput]:
    with self._lock:
        if self._max_cache_size == 0 or self._disabled:
            return None
        item = self._cache.get(key, None)
        if item is not None:
            self._hits += 1
            self._cache.move_to_end(key)
            return item.invocation_output
        self._misses += 1
        return None
```

#### حفظ النتائج
```python
def save(self, key: Union[int, str], invocation_output: BaseInvocationOutput) -> None:
    with self._lock:
        if self._max_cache_size == 0 or self._disabled or key in self._cache:
            return
        number_to_delete = len(self._cache) + 1 - self._max_cache_size
        self._delete_oldest_access(number_to_delete)
        self._cache[key] = CachedItem(
            invocation_output,
            invocation_output.model_dump_json(warnings=False, exclude_defaults=True, exclude_unset=True),
        )
```

#### حذف العناصر القديمة
```python
def _delete_oldest_access(self, number_to_delete: int) -> None:
    number_to_delete = min(number_to_delete, len(self._cache))
    for _ in range(number_to_delete):
        self._cache.popitem(last=False)
```

#### الحذف حسب المطابقة
```python
def _delete_by_match(self, to_match: str) -> None:
    with self._lock:
        if self._max_cache_size == 0:
            return
        keys_to_delete = set()
        for key, cached_item in self._cache.items():
            if to_match in cached_item.invocation_output_json:
                keys_to_delete.add(key)
        if not keys_to_delete:
            return
        for key in keys_to_delete:
            self._delete(key)
        self._invoker.services.logger.debug(
            f"Deleted {len(keys_to_delete)} cached invocation outputs for {to_match}"
        )
```

#### إنشاء المفتاح
```python
@staticmethod
def create_key(invocation: BaseInvocation) -> int:
    return hash(invocation.model_dump_json(exclude={"id"}, warnings=False))
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التعامل مع الخيوط المتعددة
```python
with self._lock:
    # العمليات الحرجة
```

### 4.2 التعامل مع التخزين المؤقت المعطل
```python
if self._max_cache_size == 0 or self._disabled:
    return None
```

### 4.3 حذف العناصر عند حذف الصور
```python
self._invoker.services.images.on_deleted(self._delete_by_match)
self._invoker.services.tensors.on_deleted(self._delete_by_match)
self._invoker.services.conditioning.on_deleted(self._delete_by_match)
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **سرعة الوصول**: OrderedDict يوفر وصولاً سريعاً.
2. **Thread-safety**: استخدام Lock للتعامل مع الخيوط المتعددة.
3. **حذف ذكي**: حذف العناصر القديمة تلقائياً.
4. **مراعاة حذف الصور**: حذف العناصر المؤقتة عند حذف الصور المرتبطة.

### نقاط الضعف
1. **استهلاك الذاكرة**: التخزين في الذاكرة قد يستهلك موارد كبيرة.
2. **لا يوجد تخزين دائم**: عند إعادة تشغيل التطبيق، تُفقد جميع البيانات المؤقتة.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Memory Cache Operation Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  get(key)                                                   │
│       │                                                     │
│       ├── Cache disabled → Return None                      │
│       │                                                     │
│       ├── Key exists → Move to end, Return item             │
│       │                                                     │
│       └── Key not exists → Increment misses, Return None   │
│                                                             │
│  save(key, output)                                          │
│       │                                                     │
│       ├── Cache disabled → Return                           │
│       │                                                     │
│       ├── Key exists → Return (no overwrite)                │
│       │                                                     │
│       └── Cache full → Delete oldest, Save new item         │
│                                                             │
│  _delete_by_match(to_match)                                 │
│       │                                                     │
│       ├── Find all keys containing to_match                 │
│       │                                                     │
│       └── Delete matching keys                              │
│                                                             │
│  create_key(invocation)                                     │
│       │                                                     │
│       └── hash(invocation.model_dump_json(exclude={"id"}))  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [Python OrderedDict](https://docs.python.org/3/library/collections.html#collections.OrderedDict)
- [Python Threading Lock](https://docs.python.org/3/library/threading.html#threading.Lock)
- [LRU Cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU))
