# توثيق ملف: config_default.py

## مسار الملف الأصلي
```
invokeai/app/services/config/config_default.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/services/config/config_default.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **نظام التكوين المركزي** (Central Configuration System) لتطبيق InvokeAI. وهو يُعرّف جميع الإعدادات القابلة للتكوين، ويعالج ترقية إصدارات التكوين، ويوفر طريقة لقراءة وكتابة ملفات التكوين بصيغة YAML.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 المكتبات المعيارية
```python
import copy
import filecmp
import locale
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional
```

### 2.2 المكتبات الخارجية
```python
import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
```
- **yaml**: قراءة وكتابة ملفات YAML.
- **pydantic**: التحقق من البيانات وتعريف النماذج.
- **pydantic_settings**: إدارة الإعدادات من متغيرات البيئة والملفات.

### 2.3 مكتبات المشروع
```python
import invokeai.configs as model_configs
from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS
from invokeai.frontend.cli.arg_parser import InvokeAIArgs
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 الثوابت
```python
INIT_FILE = Path("invokeai.yaml")
API_KEYS_FILE = Path("api_keys.yaml")
DB_FILE = Path("invokeai.db")
LEGACY_INIT_FILE = Path("invokeai.init")
PRECISION = Literal["auto", "float16", "bfloat16", "float32"]
ATTENTION_TYPE = Literal["auto", "normal", "xformers", "sliced", "torch-sdp"]
ATTENTION_SLICE_SIZE = Literal["auto", "balanced", "max", 1, 2, 3, 4, 5, 6, 7, 8]
LOG_FORMAT = Literal["plain", "color", "syslog", "legacy"]
LOG_LEVEL = Literal["debug", "info", "warning", "error", "critical"]
IMAGE_SUBFOLDER_STRATEGY = Literal["flat", "date", "type", "hash"]
CONFIG_SCHEMA_VERSION = "4.0.3"
```

### 3.2 فئة URLRegexTokenPair
```python
class URLRegexTokenPair(BaseModel):
    url_regex: str = Field(description="Regular expression to match against the URL")
    token: str = Field(description="Token to use when the URL matches the regex")

    @field_validator("url_regex")
    @classmethod
    def validate_url_regex(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex: {e}")
        return v
```
- **الهدف**: إنشاء أزواج من تعبيرات URL والنماذج.

### 3.3 فئة InvokeAIAppConfig

#### تعريف الإعدادات
```python
class InvokeAIAppConfig(BaseSettings):
    # إعدادات الشبكة
    host: str = Field(default="127.0.0.1", ...)
    port: int = Field(default=9090, ...)
    allow_origins: list[str] = Field(default=[], ...)
    allow_credentials: bool = Field(default=True, ...)
    allow_methods: list[str] = Field(default=["*"], ...)
    allow_headers: list[str] = Field(default=["*"], ...)
    ssl_certfile: Optional[Path] = Field(default=None, ...)
    ssl_keyfile: Optional[Path] = Field(default=None, ...)

    # إعدادات المسارات
    models_dir: Path = Field(default=Path("models"), ...)
    outputs_dir: Path = Field(default=Path("outputs"), ...)
    db_dir: Path = Field(default=Path("databases"), ...)
    custom_nodes_dir: Path = Field(default=Path("nodes"), ...)

    # إعدادات التسجيل
    log_handlers: list[str] = Field(default=["console"], ...)
    log_format: LOG_FORMAT = Field(default="color", ...)
    log_level: LOG_LEVEL = Field(default="info", ...)

    # إعدادات التخزين المؤقت
    max_cache_ram_gb: Optional[float] = Field(default=None, ...)
    max_cache_vram_gb: Optional[float] = Field(default=None, ...)
    enable_partial_loading: bool = Field(default=True, ...)
    keep_ram_copy_of_weights: bool = Field(default=True, ...)

    # إعدادات الجهاز
    device: str = Field(default="auto", ...)
    precision: PRECISION = Field(default="auto", ...)

    # إعدادات التوليد
    sequential_guidance: bool = Field(default=False, ...)
    attention_type: ATTENTION_TYPE = Field(default="auto", ...)
    force_tiled_decode: bool = Field(default=False, ...)

    # إعدادات الطابور
    max_queue_size: int = Field(default=10000, ...)
    clear_queue_on_startup: bool = Field(default=False, ...)

    # إعدادات النماذج
    allow_nodes: Optional[list[str]] = Field(default=None, ...)
    deny_nodes: Optional[list[str]] = Field(default=None, ...)
    node_cache_size: int = Field(default=512, ...)

    # إعدادات المستخدمين المتعددين
    multiuser: bool = Field(default=False, ...)
    strict_password_checking: bool = Field(default=False, ...)

    # مفاتيح API الخارجية
    external_alibabacloud_api_key: Optional[str] = Field(default=None, ...)
    external_gemini_api_key: Optional[str] = Field(default=None, ...)
    external_openai_api_key: Optional[str] = Field(default=None, ...)
    external_seedream_api_key: Optional[str] = Field(default=None, ...)
```

#### خصائص المسارات
```python
@property
def root_path(self) -> Path:
    if self._root:
        root = Path(self._root).expanduser().absolute()
    else:
        root = self.find_root().expanduser().absolute()
    self._root = root
    return root.resolve()

@property
def config_file_path(self) -> Path:
    resolved_path = self._resolve(self._config_file or INIT_FILE)
    return resolved_path

@property
def outputs_path(self) -> Optional[Path]:
    return self._resolve(self.outputs_dir)

@property
def db_path(self) -> Path:
    db_dir = self._resolve(self.db_dir)
    return db_dir / DB_FILE
```

### 3.4 دوال الترقية

#### migrate_v3_config_dict
```python
def migrate_v3_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    parsed_config_dict: dict[str, Any] = {}
    for _category_name, category_dict in config_dict["InvokeAI"].items():
        for k, v in category_dict.items():
            if k == "outdir":
                parsed_config_dict["outputs_dir"] = v
            if k == "max_cache_size" and "ram" not in category_dict:
                parsed_config_dict["ram"] = v
            if k == "max_vram_cache_size" and "vram" not in category_dict:
                parsed_config_dict["vram"] = v
            if k == "precision" and v == "autocast":
                parsed_config_dict["precision"] = "auto"
            # ... المزيد من الترقيات
    parsed_config_dict["schema_version"] = "4.0.0"
    return parsed_config_dict
```

### 3.5 دالة get_config()
```python
@lru_cache(maxsize=1)
def get_config() -> InvokeAIAppConfig:
    config = InvokeAIAppConfig()
    args = InvokeAIArgs.args

    if not InvokeAIArgs.did_parse:
        return config

    if root := getattr(args, "root", None):
        config._root = Path(root)
    if config_file := getattr(args, "config_file", None):
        config._config_file = Path(config_file)

    # إنشاء ملف التكوين التجريبي
    example_config = DefaultInvokeAIAppConfig()
    example_config.write_file(config.config_file_path.with_suffix(".example.yaml"), as_example=True)

    # نسخ الإعدادات القديمة
    configs_src = Path(model_configs.__path__[0])
    dest_path = config.legacy_conf_path
    dest_path.mkdir(parents=True, exist_ok=True)

    comparison = filecmp.dircmp(configs_src, dest_path)
    need_copy = any([
        comparison.left_only,
        comparison.diff_files,
        comparison.common_funny,
    ])

    if need_copy:
        shutil.copytree(configs_src, dest_path, dirs_exist_ok=True)

    # قراءة ملف التكوين
    if config.config_file_path.exists():
        config_from_file = load_and_migrate_config(config.config_file_path)
        config.update_config(config_from_file, clobber=False)
    else:
        default_config = DefaultInvokeAIAppConfig()
        default_config.write_file(config.config_file_path, as_example=False)

    # تحميل مفاتيح API الخارجية
    api_keys_from_file = load_external_api_keys(config.api_keys_file_path)
    if api_keys_from_file:
        api_keys_to_apply = {key: value for key, value in api_keys_from_file.items() if key not in env_fields_set}
        if api_keys_to_apply:
            config.update_config(api_keys_to_apply, clobber=True)

    return config
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من صحة تعبيرات URL
```python
@field_validator("url_regex")
@classmethod
def validate_url_regex(cls, v: str) -> str:
    try:
        re.compile(v)
    except re.error as e:
        raise ValueError(f"Invalid regex: {e}")
    return v
```

### 4.2 التعامل مع الترقيات
```python
if "InvokeAI" in loaded_config_dict:
    migrated = True
    loaded_config_dict = migrate_v3_config_dict(loaded_config_dict)
if loaded_config_dict["schema_version"] == "4.0.0":
    migrated = True
    loaded_config_dict = migrate_v4_0_0_to_4_0_1_config_dict(loaded_config_dict)
```

### 4.3 النسخ الاحتياطي قبل الترقية
```python
if migrated:
    shutil.copy(config_path, config_path.with_suffix(".yaml.bak"))
    try:
        migrated_config = DefaultInvokeAIAppConfig.model_validate(loaded_config_dict)
        migrated_config.write_file(config_path)
    except Exception as e:
        shutil.copy(config_path.with_suffix(".yaml.bak"), config_path)
        raise RuntimeError(f"Failed to load and migrate config") from e
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **نظام تكوين متكامل**: دعم متغيرات البيئة والملفات وأساليب سطر الأوامر.
2. **ترقية تلقائية**: التعامل مع الترقيات بين الإصدارات.
3. **typing قوي**: استخدام `Literal` للتحقق من القيم المسموح بها.
4. **Singleton**: استخدام `lru_cache` لضمان وجود نسخة واحدة فقط.

### نقاط الضعف
1. **عدد كبير من الإعدادات**: صعوبة تتبع جميع الإعدادات.
2. **ترقيات معقدة**: منطق الترقية معقد نسبياً.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Configuration System Architecture              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Environment Variables                                      │
│       │                                                     │
│       ▼                                                     │
│  CLI Arguments (InvokeAIArgs)                               │
│       │                                                     │
│       ▼                                                     │
│  invokeai.yaml                                              │
│       │                                                     │
│       ▼                                                     │
│  load_and_migrate_config()                                  │
│       │                                                     │
│       ├── migrate_v3_config_dict()                          │
│       ├── migrate_v4_0_0_to_4_0_1_config_dict()            │
│       ├── migrate_v4_0_1_to_4_0_2_config_dict()            │
│       └── migrate_v4_0_2_to_4_0_3_config_dict()            │
│       │                                                     │
│       ▼                                                     │
│  InvokeAIAppConfig                                          │
│       │                                                     │
│       ├── root_path                                         │
│       ├── config_file_path                                  │
│       ├── outputs_path                                      │
│       ├── db_path                                           │
│       └── models_path                                       │
│       │                                                     │
│       ▼                                                     │
│  get_config() [Singleton]                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [YAML Python](https://pyyaml.org/)
- [Semantic Versioning](https://semver.org/)
