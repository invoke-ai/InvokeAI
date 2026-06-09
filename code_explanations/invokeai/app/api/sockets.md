# توثيق ملف: sockets.py

## مسار الملف الأصلي
```
invokeai/app/api/sockets.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/api/sockets.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **نظام الاتصال اللحظي** (Real-time Communication System) عبر Socket.IO. وهو مسؤول عن إدارة اتصالات WebSocket، والمصادقة، والاشتراكات في الغرف، وتوجيه الأحداث للمستخدمين المتعددين.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 FastAPI و Socket.IO
```python
from fastapi import FastAPI
from pydantic import BaseModel
from socketio import ASGIApp, AsyncServer
```
- **FastAPI**: للتكامل مع تطبيق FastAPI.
- **Socket.IO**: للتعامل مع اتصالات WebSocket.
- **AsyncServer**: خادم Socket غير المتزامن.

### 2.2 خدمات المصادقة
```python
from invokeai.app.services.auth.token_service import verify_token
```

### 2.3 الأحداث
```python
from invokeai.app.services.events.events_common import (
    BatchEnqueuedEvent,
    BulkDownloadCompleteEvent,
    # ... أكثر من 30 نوع حدث
)
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 فئات الأحداث

#### QueueSubscriptionEvent
```python
class QueueSubscriptionEvent(BaseModel):
    queue_id: str
```
- بيانات الاشتراك في غرفة الطابور.

#### BulkDownloadSubscriptionEvent
```python
class BulkDownloadSubscriptionEvent(BaseModel):
    bulk_download_id: str
```
- بيانات الاشتراك في غرفة التنزيل الجماعي.

### 3.2 مجموعات الأحداث

#### QUEUE_EVENTS
```python
QUEUE_EVENTS = {
    InvocationStartedEvent,
    InvocationProgressEvent,
    InvocationCompleteEvent,
    InvocationErrorEvent,
    QueueItemStatusChangedEvent,
    BatchEnqueuedEvent,
    QueueClearedEvent,
    RecallParametersUpdatedEvent,
}
```
- أحداث مرتبطة بطابور التنفيذ.

#### MODEL_EVENTS
```python
MODEL_EVENTS = {
    DownloadCancelledEvent,
    DownloadCompleteEvent,
    DownloadErrorEvent,
    DownloadProgressEvent,
    DownloadStartedEvent,
    ModelLoadStartedEvent,
    ModelLoadCompleteEvent,
    ModelInstallDownloadProgressEvent,
    ModelInstallDownloadsCompleteEvent,
    ModelInstallStartedEvent,
    ModelInstallCompleteEvent,
    ModelInstallCancelledEvent,
    ModelInstallErrorEvent,
}
```
- أحداث مرتبطة بالنماذج والتنزيلات.

### 3.3 فئة SocketIO

#### التهيئة
```python
class SocketIO:
    _sub_queue = "subscribe_queue"
    _unsub_queue = "unsubscribe_queue"
    _sub_bulk_download = "subscribe_bulk_download"
    _unsub_bulk_download = "unsubscribe_bulk_download"

    def __init__(self, app: FastAPI):
        self._sio = AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self._app = ASGIApp(socketio_server=self._sio, socketio_path="/ws/socket.io")
        app.mount("/ws", self._app)
```

#### المصادقة
```python
async def _handle_connect(self, sid: str, environ: dict, auth: dict | None) -> bool:
    # استخراج الرمز من بيانات المصادقة أو الرؤوس
    token = None
    if auth and isinstance(auth, dict):
        token = auth.get("token")

    if not token and environ:
        headers = environ.get("HTTP_AUTHORIZATION", "")
        if headers.startswith("Bearer "):
            token = headers[7:]

    # التحقق من الرمز
    if token:
        token_data = verify_token(token)
        if token_data:
            # في وضع متعدد المستخدمين، التحقق من وجود المستخدم
            if self._is_multiuser_enabled():
                user = ApiDependencies.invoker.services.users.get(token_data.user_id)
                if user is None or not user.is_active:
                    return False

            # حفظ معلومات المستخدم
            self._socket_users[sid] = {
                "user_id": token_data.user_id,
                "is_admin": token_data.is_admin,
            }
            return True

    # في وضع مستخدم واحد، القبول كمسؤول النظام
    if self._is_multiuser_enabled():
        return False

    self._socket_users[sid] = {
        "user_id": "system",
        "is_admin": True,
    }
    return True
```

#### الاشتراك في الطابور
```python
async def _handle_sub_queue(self, sid: str, data: Any) -> None:
    queue_id = QueueSubscriptionEvent(**data).queue_id

    # التحقق من معلومات المستخدم
    if sid not in self._socket_users:
        if self._is_multiuser_enabled():
            return
        self._socket_users[sid] = {
            "user_id": "system",
            "is_admin": True,
        }

    user_id = self._socket_users[sid]["user_id"]
    is_admin = self._socket_users[sid]["is_admin"]

    # الدخول إلى غرفة الطابور
    await self._sio.enter_room(sid, queue_id)

    # الدخول إلى غرفة المستخدم
    user_room = f"user:{user_id}"
    await self._sio.enter_room(sid, user_room)

    # إذا كان مسؤولاً، الدخول إلى غرفة المسؤولين
    if is_admin:
        await self._sio.enter_room(sid, "admin")
```

#### توجيه أحداث الطابور
```python
async def _handle_queue_event(self, event: FastAPIEvent[QueueEventBase]):
    event_name, event_data = event

    # التحقق من InvocationEventBase أولاً (لأنه فئة فرعية)
    if isinstance(event_data, InvocationEventBase) and hasattr(event_data, "user_id"):
        user_room = f"user:{event_data.user_id}"
        await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room=user_room)

        # للمستخدمين المسؤولين، إرسال نسخة بدون بيانات الصورة
        if isinstance(event_data, InvocationProgressEvent):
            admin_event_data = event_data.model_copy(update={"image": None})
            await self._sio.emit(event=event_name, data=admin_event_data.model_dump(mode="json"), room="admin")
        else:
            await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room="admin")
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 رفض الاتصال بدون مصادقة
```python
if self._is_multiuser_enabled():
    logger.warning(f"Rejecting socket {sid} connection: multiuser mode is enabled")
    return False
```

### 4.2 التعامل مع المستخدمين غير النشطين
```python
user = ApiDependencies.invoker.services.users.get(token_data.user_id)
if user is None or not user.is_active:
    logger.warning(f"Rejecting socket {sid}: user {token_data.user_id} not found or inactive")
    return False
```

### 4.3 حماية بيانات الصورة
```python
if isinstance(event_data, InvocationProgressEvent):
    admin_event_data = event_data.model_copy(update={"image": None})
    await self._sio.emit(event=event_name, data=admin_event_data.model_dump(mode="json"), room="admin")
```

### 4.4 تجنب تسريب بيانات المستخدمين الآخرين
- جميع أحداث الطابور خاصة بالمستخدم المالك والمسؤولين فقط.

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **مصادقة قوية**: التحقق من JWT وحالات المستخدمين.
2. **عزل المستخدمين**: حماية بيانات المستخدمين الآخرين.
3. **دعم متعدد المستخدمين**: إمكانية التوسع لعدة مستخدمين.
4. **توجيه ذكي**: إرسال الأحداث فقط للمستخدمين المناسبين.

### نقاط الضعف
1. **تعقيد إدارة الغرف**: عدد كبير من الغرف قد يصعب إدارةه.
2. **ال依赖于 ApiDependencies**: الاعتماد على خدمة التبعيات قد يسبب مشاكل في الاختبار.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Socket.IO Communication Flow                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Client Connects                                            │
│       │                                                     │
│       ▼                                                     │
│  _handle_connect()                                          │
│       │                                                     │
│       ├── No Token → Reject (Multi-user) / Accept (Single)  │
│       │                                                     │
│       └── Valid Token → Verify User → Store user_id         │
│                                                             │
│  Client Subscribes to Queue                                 │
│       │                                                     │
│       ▼                                                     │
│  _handle_sub_queue()                                        │
│       │                                                     │
│       ├── Enter room: queue_id                              │
│       ├── Enter room: user:{user_id}                        │
│       └── Enter room: admin (if admin)                      │
│                                                             │
│  Queue Event Occurs                                         │
│       │                                                     │
│       ▼                                                     │
│  _handle_queue_event()                                      │
│       │                                                     │
│       ├── InvocationEventBase                               │
│       │     ├── Emit to user:{user_id}                      │
│       │     └── Emit to admin (strip image data)            │
│       │                                                     │
│       ├── QueueItemEventBase                                │
│       │     ├── Emit to user:{user_id}                      │
│       │     └── Emit to admin                               │
│       │                                                     │
│       └── QueueClearedEvent                                 │
│             └── Emit to queue_id (broadcast)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [Socket.IO Python Server](https://python-socketio.readthedocs.io/en/latest/server.html)
- [WebSocket Authentication](https://socket.io/docs/v4/authentication/)
- [Room Management](https://socket.io/docs/v4/rooms/)
