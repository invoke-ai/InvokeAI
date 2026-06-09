# توثيق ملف: devices.py

## مسار الملف الأصلي
```
invokeai/backend/util/devices.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/backend/util/devices.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **طبقة التجريد للأجهزة** (Device Abstraction Layer) في InvokeAI. وهو يوفر واجهة موحدة للتعامل مع أجهزة الحوسبة المختلفة (CPU، CUDA، MPS) وتحديد الدقة المناسبة لكل جهاز.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 المكتبات المعيارية
```python
from typing import Dict, Literal, Optional, Union
```

### 2.2 PyTorch
```python
import torch
```

### 2.3 مكتبات المشروع
```python
from invokeai.app.services.config.config_default import get_config
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 الثوابت
```python
TorchPrecisionNames = Literal["float32", "float16", "bfloat16"]
CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda")
MPS_DEVICE = torch.device("mps")
```

### 3.2 الدوال القديمة (Deprecated)
```python
@deprecated("Use TorchDevice.choose_torch_dtype() instead.")
def choose_precision(device: torch.device) -> TorchPrecisionNames:
    torch_dtype = TorchDevice.choose_torch_dtype(device)
    return PRECISION_TO_NAME[torch_dtype]

@deprecated("Use TorchDevice.choose_torch_device() instead.")
def choose_torch_device() -> torch.device:
    return TorchDevice.choose_torch_device()

@deprecated("Use TorchDevice.choose_torch_dtype() instead.")
def torch_dtype(device: torch.device) -> torch.dtype:
    return TorchDevice.choose_torch_dtype(device)
```

### 3.3 قاموس الدقة
```python
NAME_TO_PRECISION: Dict[TorchPrecisionNames, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
PRECISION_TO_NAME: Dict[torch.dtype, TorchPrecisionNames] = {v: k for k, v in NAME_TO_PRECISION.items()}
```

### 3.4 فئة TorchDevice

#### اختيار الجهاز
```python
@classmethod
def choose_torch_device(cls) -> torch.device:
    app_config = get_config()
    if app_config.device != "auto":
        device = torch.device(app_config.device)
    elif torch.cuda.is_available():
        device = CUDA_DEVICE
    elif torch.backends.mps.is_available():
        device = MPS_DEVICE
    else:
        device = CPU_DEVICE
    return cls.normalize(device)
```

#### اختيار الدقة
```python
@classmethod
def choose_torch_dtype(cls, device: Optional[torch.device] = None) -> torch.dtype:
    device = device or cls.choose_torch_device()
    config = get_config()
    if device.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(device)
        if "GeForce GTX 1660" in device_name or "GeForce GTX 1650" in device_name:
            return cls._to_dtype("float32")
        elif config.precision == "auto":
            return cls._to_dtype("float16")
        else:
            return cls._to_dtype(config.precision)
    elif device.type == "mps" and torch.backends.mps.is_available():
        if config.precision == "auto":
            return cls._to_dtype("float16")
        else:
            return cls._to_dtype(config.precision)
    return cls._to_dtype("float32")
```

#### تفريغ الذاكرة
```python
@classmethod
def empty_cache(cls) -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

#### اختيار دقة bfloat16 الآمنة
```python
@classmethod
def choose_bfloat16_safe_dtype(cls, device: Optional[torch.device] = None) -> torch.dtype:
    device = device or cls.choose_torch_device()
    try:
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        return torch.bfloat16
    except TypeError:
        if device.type == "cuda":
            return torch.float16
        return torch.float32
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التعامل مع الأجهزة غير المدعومة
```python
try:
    torch.tensor([1.0], dtype=torch.bfloat16, device=device)
    return torch.bfloat16
except TypeError:
    if device.type == "cuda":
        return torch.float16
    return torch.float32
```

### 4.2 التعامل مع الأجهزة القديمة
```python
if "GeForce GTX 1660" in device_name or "GeForce GTX 1650" in device_name:
    return cls._to_dtype("float32")
```

### 4.3 تطبيع الجهاز
```python
@classmethod
def normalize(cls, device: Union[str, torch.device]) -> torch.device:
    device = torch.device(device)
    if device.index is None and device.type == "cuda" and torch.cuda.is_available():
        device = torch.device(device.type, torch.cuda.current_device())
    return device
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **تجريد الأجهزة**: واجهة موحدة لأجهزة مختلفة.
2. **كشف تلقائي**: تحديد الأجهزة المدعومة تلقائياً.
3. **مراعاة الأجهزة القديمة**: التعامل مع الأجهزة ذات الدعم المحدود.
4. **تفريغ الذاكرة**: دعم تفريغ ذاكرة GPU.

### نقاط الضعف
1. **ال依赖 على التكوين**: الاعتماد على `get_config()` في كل استدعاء.
2. **لا يوجد تخزين مؤقت**: يتم حساب كل شيء في كل مرة.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              TorchDevice Selection Flow                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  get_config().device                                        │
│       │                                                     │
│       ├── "auto"                                            │
│       │     │                                               │
│       │     ├── torch.cuda.is_available() → CUDA            │
│       │     │                                               │
│       │     ├── torch.backends.mps.is_available() → MPS     │
│       │     │                                               │
│       │     └── else → CPU                                  │
│       │                                                     │
│       └── specific device → Use as-is                       │
│                                                             │
│  normalize(device)                                          │
│       │                                                     │
│       └── Add device index for CUDA                         │
│                                                             │
│  choose_torch_dtype(device)                                 │
│       │                                                     │
│       ├── CUDA                                              │
│       │     ├── GTX 1660/1650 → float32                     │
│       │     ├── auto → float16                              │
│       │     └── specific → Use as-is                        │
│       │                                                     │
│       ├── MPS                                               │
│       │     ├── auto → float16                              │
│       │     └── specific → Use as-is                        │
│       │                                                     │
│       └── CPU → float32                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [PyTorch dtype](https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype)
