# المستند الثالث: بروتوكولات الاتصال وتدفق البيانات
## Network Communication & Data Flow Architecture

```
invokeai/
├── docs/
│   └── technical_deep_dive/
│       └── network_communication.md  <-- هذا الملف
```

---

## ملخص البحث

يُقدّم هذا المستند تحليلاً شاملاً لهندسة الاتصال بين مكونات مشروع InvokeAI، بما في ذلك REST API، WebSocket، وHTTP Proxy. يُركّز التحليل على الآليات البرمجية التي تُمكّن الاتصال السلس بين الواجهة الأمامية (Frontend) والخلفية (Backend).

---

## أولاً: نظرة عامة على معمارية الاتصال

```
┌─────────────────────────────────────────────────────────────────┐
│                    معمارية الاتصال في InvokeAI                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │   Frontend (React)  │         │   Backend (FastAPI)  │       │
│  │   Port: 5173        │◄───────►│   Port: 9090         │       │
│  │                     │         │                      │       │
│  │  ┌───────────────┐  │  HTTP   │  ┌───────────────┐  │       │
│  │  │   REST API    │  │◄───────►│  │   API Router   │  │       │
│  │  │   (RTK Query) │  │         │  │   (FastAPI)    │  │       │
│  │  └───────────────┘  │         │  └───────────────┘  │       │
│  │                     │         │                      │       │
│  │  ┌───────────────┐  │   WS    │  ┌───────────────┐  │       │
│  │  │   Socket.IO   │  │◄───────►│  │   Socket.IO   │  │       │
│  │  │   Client      │  │         │  │   Server      │  │       │
│  │  └───────────────┘  │         │  └───────────────┘  │       │
│  │                     │         │                      │       │
│  └─────────────────────┘         └─────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ثانياً: REST API (FastAPI Backend)

### 2.1. بنية API الأساسية

يُقدّم ملف `invokeai/app/api_app.py` ([api_app.py:75-192](api_app.py#L75-L192)) التطبيق الرئيسي:

```python
app = FastAPI(
    title="Invoke - Community Edition",
    docs_url=None,
    redoc_url=None,
    separate_input_output_schemas=False,
    lifespan=lifespan,
)

# تسجيل مسارات API
app.include_router(auth.auth_router, prefix="/api")
app.include_router(model_manager.model_manager_router, prefix="/api")
app.include_router(session_queue.session_queue_router, prefix="/api")
app.include_router(images.images_router, prefix="/api")
# ... باقي المسارات
```

### 2.2. مسارات API الرئيسية

| المسار | الملف | الوظيفة |
|---|---|---|
| `/api/v1/auth` | `auth.py` | المصادقة والتفويض |
| `/api/v1/models` | `model_manager.py` | إدارة النماذج |
| `/api/v1/queue` | `session_queue.py` | طابور المعالجة |
| `/api/v1/images` | `images.py` | إدارة الصور |
| `/api/v1/boards` | `boards.py` | إدارة الألباب |
| `/api/v1/workflows` | `workflows.py` | سير العمل |

### 2.3. معالجة طلبات التوليد

يُقدّم ملف `invokeai/app/api/routers/session_queue.py` ([session_queue.py:86-106](session_queue.py#L86-L106)) معالجة الدفعات:

```python
@session_queue_router.post(
    "/{queue_id}/enqueue_batch",
    operation_id="enqueue_batch",
    responses={201: {"model": EnqueueBatchResult}},
)
async def enqueue_batch(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    batch: Batch = Body(description="Batch to process"),
    prepend: bool = Body(default=False, description="Whether or not to prepend this batch"),
) -> EnqueueBatchResult:
    """Processes a batch and enqueues the output graphs for execution."""
    try:
        return await ApiDependencies.invoker.services.session_queue.enqueue_batch(
            queue_id=queue_id, batch=batch, prepend=prepend, user_id=current_user.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
```

**هيكل الطلب:**

```json
{
  "batch": {
    "graph": {
      "nodes": {
        "1": {
          "type": "txt2img",
          "prompt": "A luxury minimalist room",
          "negative_prompt": "ugly, blurry",
          "model": "sd-1-5",
          "steps": 25,
          "cfg_scale": 7.5,
          "width": 512,
          "height": 512,
          "seed": 12345
        }
      },
      "edges": []
    }
  },
  "prepend": false
}
```

### 2.4. إدارة النماذج

يُقدّم ملف `invokeai/app/api/routers/model_manager.py` مسارات لإدارة النماذج:

```python
@model_manager_router.get(
    "/v2/models/",
    operation_id="list_models",
    responses={200: {"model": list[AnyModelConfig]}},
)
async def list_models(
    base: Optional[BaseModelType] = Query(default=None),
    type: Optional[ModelType] = Query(default=None),
) -> list[AnyModelConfig]:
    """List all models with optional filtering."""
    # ...
```

**المسارات الرئيسية للنماذج:**

| الطريقة | المسار | الوظيفة |
|---|---|---|
| `GET` | `/v2/models/` | قائمة النماذج |
| `GET` | `/v2/models/{key}` | تفاصيل نموذج |
| `POST` | `/v2/models/install` | تثبيت نموذج |
| `DELETE` | `/v2/models/{key}` | حذف نموذج |
| `GET` | `/v2/model_configs/` | قائمة التكوينات |

---

## ثالثاً: WebSocket (Socket.IO)

### 3.1. خادم Socket.IO

يُقدّم ملف `invokeai/app/api/sockets.py` ([sockets.py:91-118](sockets.py#L91-L118)) خادم WebSocket:

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
        
        # معالجة الأحداث
        self._sio.on("connect", handler=self._handle_connect)
        self._sio.on("disconnect", handler=self._handle_disconnect)
        self._sio.on(self._sub_queue, handler=self._handle_sub_queue)
```

### 3.2. أحداث WebSocket الرئيسية

**أحداث التوليد (Queue Events):**

```python
QUEUE_EVENTS = {
    InvocationStartedEvent,      # بدء المعالجة
    InvocationProgressEvent,     # تقدم المعالجة
    InvocationCompleteEvent,     # اكتمال المعالجة
    InvocationErrorEvent,        # خطأ في المعالجة
    QueueItemStatusChangedEvent, # تغيير حالة العنصر
    BatchEnqueuedEvent,          # إضافة دفعة
    QueueClearedEvent,           # مسح الطابور
}
```

**أحداث النماذج (Model Events):**

```python
MODEL_EVENTS = {
    DownloadStartedEvent,        # بدء التنزيل
    DownloadProgressEvent,       # تقدم التنزيل
    DownloadCompleteEvent,       # اكتمال التنزيل
    DownloadErrorEvent,          # خطأ في التنزيل
    ModelLoadStartedEvent,       # بدء تحميل النموذج
    ModelLoadCompleteEvent,      # اكتمال تحميل النموذج
    ModelInstallStartedEvent,    # بدء التثبيت
    ModelInstallCompleteEvent,   # اكتمال التثبيت
}
```

### 3.3. معالجة الأحداث

يُقدّم ملف `invokeai/app/api/sockets.py` ([sockets.py:263-343](sockets.py#L263-L343)) معالجة أحداث الطابور:

```python
async def _handle_queue_event(self, event: FastAPIEvent[QueueEventBase]):
    event_name, event_data = event
    
    # الأحداث الخاصة بالمستخدم فقط
    if isinstance(event_data, InvocationEventBase) and hasattr(event_data, "user_id"):
        user_room = f"user:{event_data.user_id}"
        
        # إرسال للمستخدم
        await self._sio.emit(
            event=event_name,
            data=event_data.model_dump(mode="json"),
            room=user_room
        )
        
        # إرسال للمدير مع حذف بيانات الصورة
        if isinstance(event_data, InvocationProgressEvent):
            admin_event_data = event_data.model_copy(update={"image": None})
            await self._sio.emit(
                event=event_name,
                data=admin_event_data.model_dump(mode="json"),
                room="admin"
            )
```

### 3.4. نظام الغرف (Rooms)

```
┌─────────────────────────────────────────────────────────────┐
│                    نظام الغرف في Socket.IO                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Room: queue_id                    │   │
│  │  ├── جميع أحداث الطابور                            │   │
│  │  └── جميع المستخدمين المشتركين                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Room: user:{user_id}               │   │
│  │  ├── أحداث المستخدم الخاصة فقط                     │   │
│  │  └── حماية الخصوصية                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Room: admin                      │   │
│  │  ├── جميع الأحداث (باستثناء صور التقدم)           │   │
│  │  └── للمديرين فقط                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## رابعاً: عميل Socket.IO (Frontend)

### 4.1. تهيئة الاتصال

يُقدّم ملف `invokeai/frontend/web/src/services/events/useSocketIO.ts` ([useSocketIO.ts:23-82](useSocketIO.ts#L23-L82)) تهيئة الاتصال:

```typescript
export const useSocketIO = () => {
  const socketUrl = useMemo(() => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${wsProtocol}://${window.location.host}`;
  }, []);

  const socketOptions = useMemo(() => {
    const token = localStorage.getItem('auth_token');
    const options: Partial<ManagerOptions & SocketOptions> = {
      timeout: 60000,
      path: '/ws/socket.io',
      autoConnect: false,
      forceNew: true,
      auth: token ? { token } : undefined,
      extraHeaders: token
        ? { Authorization: `Bearer ${token}` }
        : undefined,
    };
    return options;
  }, []);

  useEffect(() => {
    const socket: AppSocket = io(socketUrl, socketOptions);
    $socket.set(socket);
    setEventListeners({ socket, store, setIsConnected: $isConnected.set });
    socket.connect();
    // ...
  }, [socketOptions, socketUrl, store]);
};
```

### 4.2. مستمعي الأحداث

يُقدّم ملف `invokeai/frontend/web/src/services/events/setEventListeners.tsx` ([setEventListeners.tsx:72-935](setEventListeners.tsx#L72-L935)) مستمعي الأحداث:

```typescript
export const setEventListeners = ({ socket, store, setIsConnected }: SetEventListenersArg) => {
  const { dispatch, getState } = store;

  // عند الاتصال
  socket.on('connect', () => {
    setIsConnected(true);
    dispatch(socketConnected());
    socket.emit('subscribe_queue', { queue_id: 'default' });
    socket.emit('subscribe_bulk_download', { bulk_download_id: 'default' });
  });

  // بدء المعالجة
  socket.on('invocation_started', (data) => {
    const { invocation_source_id, invocation } = data;
    const nes = $nodeExecutionStates.get()[invocation_source_id];
    const updatedNodeExecutionState = getUpdatedNodeExecutionStateOnInvocationStarted(
      nes, data, completedInvocationKeysByItemId
    );
    if (updatedNodeExecutionState) {
      upsertExecutionState(updatedNodeExecutionState.nodeId, updatedNodeExecutionState);
    }
  });

  // تقدم المعالجة
  socket.on('invocation_progress', (data) => {
    const { invocation_source_id, percentage, message } = data;
    $lastProgressEvent.set(data);
    
    if (origin === 'workflows') {
      const nes = $nodeExecutionStates.get()[invocation_source_id];
      const updatedNodeExecutionState = getUpdatedNodeExecutionStateOnInvocationProgress(
        nes, data, completedInvocationKeysByItemId
      );
      if (updatedNodeExecutionState) {
        upsertExecutionState(updatedNodeExecutionState.nodeId, updatedNodeExecutionState);
      }
    }
  });

  // اكتمال المعالجة
  socket.on('invocation_complete', onInvocationComplete);
  
  // خطأ في المعالجة
  socket.on('invocation_error', (data) => {
    const { invocation_source_id, invocation } = data;
    const nes = $nodeExecutionStates.get()[invocation_source_id];
    const updatedNodeExecutionState = getUpdatedNodeExecutionStateOnInvocationError(nes, data);
    if (updatedNodeExecutionState) {
      upsertExecutionState(updatedNodeExecutionState.nodeId, updatedNodeExecutionState);
    }
  });
};
```

### 4.3. تتبع حالة العقدة

يُقدّم ملف `invokeai/frontend/web/src/services/events/nodeExecutionState.ts` ([nodeExecutionState.ts](nodeExecutionState.ts)) تتبع حالة كل عقدة:

```typescript
export type NodeExecutionState = {
  nodeId: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  progress: number | null;
  progressImage: ProgressImage | null;
  outputs: unknown[];
  error: string | null;
};
```

---

## خامساً: HTTP Proxy (Vite Dev Server)

### 5.1. تكوين الـ Proxy

يُقدّم ملف `invokeai/frontend/web/vite.config.mts` ([vite.config.mts:22-40](vite.config.mts#L22-L40)) تكوين الـ Proxy:

```typescript
server: {
  proxy: {
    // WebSocket
    '/ws/socket.io': {
      target: 'ws://127.0.0.1:9090',
      ws: true,
    },
    // OpenAPI Schema
    '/openapi.json': {
      target: 'http://127.0.0.1:9090/openapi.json',
      rewrite: (path) => path.replace(/^\/openapi.json/, ''),
      changeOrigin: true,
    },
    // REST API
    '/api/': {
      target: 'http://127.0.0.1:9090/api/',
      rewrite: (path) => path.replace(/^\/api/, ''),
      changeOrigin: true,
    },
  },
  host: '0.0.0.0',
},
```

### 5.2. آلية عمل الـ Proxy

```
┌─────────────────────────────────────────────────────────────┐
│                    آليات عمل HTTP Proxy                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. طلب WebSocket (ws://localhost:5173/ws/socket.io)       │
│     ├── Vite Proxy يحوله إلى: ws://127.0.0.1:9090/ws/... │
│     └── الاتصال متواصل عبر TCP                             │
│                                                              │
│  2. طلب REST API (http://localhost:5173/api/v1/...)        │
│     ├── Vite Proxy يحوله إلى: http://127.0.0.1:9090/...  │
│     └── يحذف prefix /api                                   │
│                                                              │
│  3. طلب OpenAPI (http://localhost:5173/openapi.json)       │
│     ├── Vite Proxy يحوله إلى: http://127.0.0.1:9090/...  │
│     └── يحذف prefix /openapi.json                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3. حل مشاكل CORS

**المشكلة:** المتصفح يمنع الطلبات عبر Origin مختلفة.

**الحل 1: Vite Proxy (للتطوير)**

```typescript
// في vite.config.mts
server: {
  proxy: {
    '/api/': {
      target: 'http://127.0.0.1:9090',
      changeOrigin: true,  // يحول Origin
    }
  }
}
```

**الحل 2: CORS Middleware (للإنتاج)**

```python
# في api_app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.allow_origins,
    allow_credentials=app_config.allow_credentials,
    allow_methods=app_config.allow_methods,
    allow_headers=app_config.allow_headers,
    expose_headers=["X-Refreshed-Token"],
)
```

### 5.4. تدفق الاتصال الكامل

```
┌─────────────────────────────────────────────────────────────────┐
│                    تدفق الاتصال الكامل                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. المستخدم يضغط "Generate"                                    │
│     └── Frontend: dispatch(enqueueBatch(batch))                 │
│                                                                  │
│  2. RTK Query يرسل طلب POST                                    │
│     └── POST http://localhost:5173/api/v1/queue/default/enqueue │
│                                                                  │
│  3. Vite Proxy يحول الطلب                                       │
│     └── POST http://127.0.0.1:9090/v1/queue/default/enqueue    │
│                                                                  │
│  4. FastAPI يستقبل الطلب                                        │
│     └── session_queue.enqueue_batch()                           │
│                                                                  │
│  5. Backend يرسل استجابة                                        │
│     └── 201 Created: { item_id: 123, batch_id: "abc" }        │
│                                                                  │
│  6. Backend يبدأ المعالجة                                       │
│     └── Socket.IO: emit('invocation_started', {...})            │
│                                                                  │
│  7. Frontend يستقبل الحدث                                       │
│     └── socket.on('invocation_started', handleStarted)          │
│                                                                  │
│  8. Backend يرسل تحديثات التقدم                                 │
│     └── Socket.IO: emit('invocation_progress', {...})           │
│                                                                  │
│  9. Frontend يعرض شريط التقدم                                   │
│     └── setProgress(percentage)                                 │
│                                                                  │
│  10. Backend يكتمل المعالجة                                     │
│      └── Socket.IO: emit('invocation_complete', {...})          │
│                                                                  │
│  11. Frontend يعرض الصورة النهائية                               │
│      └── setImage(resultImage)                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## سادساً: أحداث التقدمReal-time

### 6.1. هيكل حدث التقدم

```python
class InvocationProgressEvent(QueueEventBase):
    """حدث تقدم المعالجة"""
    item_id: int
    invocation_source_id: str
    invocation: InvocationBase
    percentage: float  # 0.0 إلى 1.0
    message: str | None
    image: str | None  # صورة مصغرة اختيارية
```

### 6.2. حساب نسبة التقدم

```python
# في diffusers_pipeline.py
for i, t in enumerate(self.progress_bar(timesteps)):
    # نسبة التقدم = الخطوة الحالية / إجمالي الخطوات
    percentage = (i + 1) / len(timesteps)
    
    # إرسال حدث التقدم
    callback(PipelineIntermediateState(
        step=i + 1,
        order=self.scheduler.order,
        total_steps=len(timesteps),
        timestep=int(t),
        latents=latents,
    ))
```

### 6.3. عرض التقدم في الواجهة

```typescript
// في setEventListeners.tsx
socket.on('invocation_progress', (data) => {
  const { percentage, message } = data;
  
  // تحديث حالة العقدة
  if (origin === 'workflows') {
    const nes = $nodeExecutionStates.get()[invocation_source_id];
    const updatedNodeExecutionState = getUpdatedNodeExecutionStateOnInvocationProgress(
      nes, data, completedInvocationKeysByItemId
    );
    if (updatedNodeExecutionState) {
      upsertExecutionState(updatedNodeExecutionState.nodeId, updatedNodeExecutionState);
    }
  }
  
  // تحديث آخر حدث تقدم
  $lastProgressEvent.set(data);
});
```

---

## سابعاً: إدارة الحالة (State Management)

### 7.1. Redux Toolkit

يُستخدم Redux Toolkit لإدارة الحالة في الواجهة الأمامية:

```typescript
// في services/api/index.ts
export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({ baseUrl: '/api/v1' }),
  tagTypes: [
    'ModelConfig',
    'SessionQueueItem',
    'BatchStatus',
    'Image',
    // ...
  ],
  endpoints: (builder) => ({
    // نقاط النهاية
  }),
});
```

### 7.2. RTK Query Cache

```
┌─────────────────────────────────────────────────────────────┐
│                    RTK Query Cache Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. طلب أولي: GET /api/v1/models/                           │
│     ├── Cache Miss → إرسال طلب HTTP                         │
│     └── حفظ النتيجة في Cache                                │
│                                                              │
│  2. طلب متكرر: GET /api/v1/models/                          │
│     ├── Cache Hit → إرجاع النتيجة المحفوظة                  │
│     └── لا حاجة لطلب HTTP                                   │
│                                                              │
│  3. تغيير البيانات: POST /api/v1/models/install             │
│     ├── Cache Invalidation → حذف Cache القديم               │
│     └── إعادة جلب البيانات                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.3. تحديث Cache عبر Socket.IO

```typescript
// في setEventListeners.tsx
socket.on('queue_item_status_changed', (data) => {
  const { item_id, status } = data;
  
  // تحديث Cache مباشرة
  dispatch(
    queueApi.util.updateQueryData('getQueueItem', item_id, (draft) => {
      draft.status = status;
      // ...
    })
  );
  
  // إبطال Cache المتعلق
  dispatch(queueApi.util.invalidateTags([
    'SessionQueueStatus',
    'CurrentSessionQueueItem',
    // ...
  ]));
});
```

---

## ثامناً: الأمان والمصادقة

### 8.1. نظام JWT

```python
# في api_app.py
class SlidingWindowTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # تحديث التوكن في كل طلب mutating
        if response.status_code < 400 and request.method in ("POST", "PUT", "PATCH", "DELETE"):
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                token_data = verify_token(token)
                if token_data is not None:
                    # إنشاء توكن جديد
                    new_token = create_access_token(token_data, expires_delta)
                    response.headers["X-Refreshed-Token"] = new_token
        
        return response
```

### 8.2. مصادقة Socket.IO

```python
# في sockets.py
async def _handle_connect(self, sid: str, environ: dict, auth: dict | None) -> bool:
    token = None
    if auth and isinstance(auth, dict):
        token = auth.get("token")
    
    if token:
        token_data = verify_token(token)
        if token_data:
            # حفظ معلومات المستخدم
            self._socket_users[sid] = {
                "user_id": token_data.user_id,
                "is_admin": token_data.is_admin,
            }
            return True
    
    # في وضع المستخدم الواحد، قبول بدون مصادقة
    if not self._is_multiuser_enabled():
        self._socket_users[sid] = {
            "user_id": "system",
            "is_admin": True,
        }
        return True
    
    return False
```

### 8.3. حماية المستخدمين المتعددين

```python
async def _handle_sub_queue(self, sid: str, data: Any) -> None:
    queue_id = QueueSubscriptionEvent(**data).queue_id
    user_id = self._socket_users[sid]["user_id"]
    is_admin = self._socket_users[sid]["is_admin"]
    
    # إضافة لغرفة الطابور
    await self._sio.enter_room(sid, queue_id)
    
    # إضافة لغرفة المستخدم
    user_room = f"user:{user_id}"
    await self._sio.enter_room(sid, user_room)
    
    # إضافة لغرفة المدير
    if is_admin:
        await self._sio.enter_room(sid, "admin")
```

---

## تاسعاً: ملخص البروتوكولات

| البروتوكول | الاستخدام | الميزة الرئيسية |
|---|---|---|
| **HTTP/REST** | CRUD Operations | بساطة + Caching |
| **WebSocket** | Real-time Updates | تحديثات فورية |
| **HTTP Proxy** | Development | حل CORS |
| **JWT** | Authentication | أمان + مرونة |

---

## عاشراً: استنتاجات أكاديمية

### 10.1. مزايا التصميم

1. **فصل الاهتمامات:** REST للعمليات، WebSocket للتحديثات
2. **أمان متعدد المستويات:** JWT + Rooms + User Isolation
3. **.performance:** Cache + Proxy + Optimistic Updates
4. **قابلية التوسع:** تصميم معماري يدعم المستخدمين المتعددين

### 10.2. تحديات تقنية

1. **تعقيد WebSocket:** إدارة الغرف والمستخدمين
2. **تن syncing Cache:** بين REST و WebSocket
3. **أمان الاتصال:** حماية البيانات الحساسة
4. **性能调节:** تحسين استهلاك الموارد

### 10.3. توصيات للتحسين

1. **استخدام GraphQL:** بدلاً من REST للعمليات المعقدة
2. **Server-Sent Events:** بديل أبسط لـ WebSocket
3. **CDN:** لتوزيع الملفات الثابتة
4. **Rate Limiting:** حماية من الطلبات المفرطة

---

## المراجع التقنية

1. FastAPI Documentation - https://fastapi.tiangolo.com/
2. Socket.IO Documentation - https://socket.io/docs/v4/
3. Vite Documentation - https://vitejs.dev/
4. Redux Toolkit - https://redux-toolkit.js.org/
5. RTK Query - https://redux-toolkit.js.org/rtk-query/overview

---

*آخر تحديث: يونيو 2026*
*المؤلف: قسم أبحاث الذكاء الاصطناعي*
