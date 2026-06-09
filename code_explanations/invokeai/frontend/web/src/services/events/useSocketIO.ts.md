# توثيق ملف: useSocketIO.ts

## مسار الملف الأصلي
```
invokeai/frontend/web/src/services/events/useSocketIO.ts
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/frontend/web/src/services/events/useSocketIO.ts.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **خطاف React** (React Hook) لتهيئة اتصال Socket.IO مع الخادم. وهو مسؤول عن إنشاء الاتصال، وإعداد مستمعي الأحداث، وإدارة حالة الاتصال.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 React
```typescript
import { useEffect, useMemo } from 'react';
```

### 2.2 Socket.IO Client
```typescript
import { io } from 'socket.io-client';
import type { ManagerOptions, SocketOptions } from 'socket.io-client';
```

### 2.3 مكتبات المشروع
```typescript
import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectQueueStatus } from 'services/api/endpoints/queue';
import { setEventListeners } from 'services/events/setEventListeners';
import type { AppSocket } from 'services/events/types';
import { $isConnected, $lastProgressEvent, $socket } from './stores';
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 إنشاء عنوان الاتصال
```typescript
const socketUrl = useMemo(() => {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${wsProtocol}://${window.location.host}`;
}, []);
```
- تحديد البروتوكول (ws/wss) بناءً على بروتوكول الصفحة.

### 3.2 خيارات الاتصال
```typescript
const socketOptions = useMemo(() => {
  const token = localStorage.getItem('auth_token');
  const options: Partial<ManagerOptions & SocketOptions> = {
    timeout: 60000,
    path: '/ws/socket.io',
    autoConnect: false,
    forceNew: true,
    auth: token ? { token } : undefined,
    extraHeaders: token ? { Authorization: `Bearer ${token}` } : undefined,
  };
  return options;
}, []);
```

### 3.3 تهيئة الاتصال
```typescript
useEffect(() => {
  const socket: AppSocket = io(socketUrl, socketOptions);
  $socket.set(socket);

  setEventListeners({ socket, store, setIsConnected: $isConnected.set });

  socket.connect();

  // مراقبة حالة الطابور
  const unsubscribeQueueStatusListener = store.subscribe(() => {
    const queueStatusData = selectQueueStatus(store.getState()).data;
    if (!queueStatusData || queueStatusData.queue.in_progress === 0) {
      $lastProgressEvent.set(null);
    }
  });

  return () => {
    unsubscribeQueueStatusListener();
    socket.disconnect();
  };
}, [socketOptions, socketUrl, store]);
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من_singleton
```typescript
useAssertSingleton('useSocketIO');
```
- التأكد من أن الخطاف يستخدم مرة واحدة فقط.

### 4.2 المصادقة
```typescript
const token = localStorage.getItem('auth_token');
auth: token ? { token } : undefined,
extraHeaders: token ? { Authorization: `Bearer ${token}` } : undefined,
```

### 4.3 تنظيف الاتصال
```typescript
return () => {
  socket.disconnect();
};
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **استخدام useMemo**: تجنب إعادة حساب العنوان والخيارات.
2. **تنظيف الموارد**: قطع الاتصال عند إلغاء تحميل المكون.
3. **مراقبة الحالة**: تصفير أحداث التقدم عند اكتمال الطابور.

### نقاط الضعف
1. **لا يوجد إعادة اتصال تلقائية**: إذا انقطع الاتصال، يجب إعادة التحميل يدوياً.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Socket.IO Connection Flow                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  useSocketIO()                                              │
│       │                                                     │
│       ├── useMemo: socketUrl                                │
│       │     └── ws://host or wss://host                     │
│       │                                                     │
│       ├── useMemo: socketOptions                            │
│       │     ├── timeout: 60000                              │
│       │     ├── path: /ws/socket.io                         │
│       │     ├── autoConnect: false                          │
│       │     ├── auth: { token }                             │
│       │     └── extraHeaders: { Authorization: Bearer }     │
│       │                                                     │
│       └── useEffect (Mount)                                 │
│             │                                               │
│             ├── io(socketUrl, socketOptions)                │
│             │     └── Create socket connection              │
│             │                                               │
│             ├── $socket.set(socket)                         │
│             │                                               │
│             ├── setEventListeners(socket, store)            │
│             │     └── Set up event handlers                 │
│             │                                               │
│             ├── socket.connect()                            │
│             │     └── Establish WebSocket connection        │
│             │                                               │
│             ├── store.subscribe()                           │
│             │     └── Monitor queue status                  │
│             │                                               │
│             └── Return cleanup                              │
│                   └── socket.disconnect()                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [Socket.IO Client](https://socket.io/docs/v4/client-installation/)
- [React Hooks](https://react.dev/reference/react/hooks)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket_API)
