# Canvas Multi-Instance Implementation Plan

## Overview
Transform the InvokeAI canvas from a singleton architecture to support multiple canvas instances within the canvas tab. Each instance will have independent state, tools, and generation capabilities while sharing the same UI framework.

## Architecture Summary

### Current State
- Single canvas instance with singleton Redux slice (`canvasSlice`)
- Global `CanvasManager` instance stored in `$canvasManager` atom
- Direct component coupling to global canvas state
- Single workspace panel in dockview layout

### Target State
- Multiple canvas instances as dockview panels within main panel
- Redux slice supporting multiple canvas states with active instance tracking
- Canvas manager registry with instance lifecycle management
- Context-based API abstraction for components
- Per-instance readiness and generation

## Implementation Phases

## Phase 1: Redux State Architecture Refactoring

To achieve isolated undo/redo histories, the Redux architecture will be split into two main parts: a `canvasInstanceSlice` that manages the state for a single canvas, and a `canvasesSlice` that acts as a router, managing a collection of undoable instances.

### 1.1 Create `canvasInstanceSlice`

**File**: `src/features/controlLayers/store/canvasInstanceSlice.ts` (new)

This new slice will contain all the drawing-related reducers from the original `canvasSlice`. Its reducer will be wrapped with `redux-undo` to create a single, undoable history.

```typescript
import { createSlice } from '@reduxjs/toolkit';
import undoable, { type UndoableOptions } from 'redux-undo';
import { getInitialCanvasState, type CanvasState } from './types';

// All existing drawing reducers (rasterLayerAdded, etc.) go here
const reducers = { /* ... */ };

const canvasInstanceSlice = createSlice({
  name: 'canvasInstance',
  initialState: getInitialCanvasState(),
  reducers: reducers,
});

const undoableConfig: UndoableOptions<CanvasState> = { limit: 64 };

// Export the undoable reducer for a single instance
export const undoableCanvasInstanceReducer = undoable(canvasInstanceSlice.reducer, undoableConfig);
export const instanceActions = canvasInstanceSlice.actions;
```

### 1.2 Create `canvasesSlice` as a Router

**File**: `src/features/controlLayers/store/canvasesSlice.ts` (renamed from `canvasSlice.ts`)

This slice manages the dictionary of canvas instances. It uses `extraReducers` to act as a router, forwarding actions from UI components to the correct `undoableCanvasInstanceReducer` based on `canvasId`.

```typescript
import { createSlice, type PayloadAction, isAnyOf } from '@reduxjs/toolkit';
import { undoableCanvasInstanceReducer, instanceActions } from './canvasInstanceSlice';
import { type Undoable } from 'redux-undo';
import { type CanvasState } from './types';

interface CanvasesState {
  instances: Record<string, Undoable<CanvasState>>;
  activeInstanceId: string | null;
}

const initialCanvasesState: CanvasesState = { instances: {}, activeInstanceId: null };

export const canvasesSlice = createSlice({
  name: 'canvases',
  initialState: initialCanvasesState,
  reducers: {
    canvasInstanceAdded: (state, action: PayloadAction<{ canvasId: string }>) => {
      const { canvasId } = action.payload;
      state.instances[canvasId] = undoableCanvasInstanceReducer(undefined, { type: '@@INIT' });
    },
    canvasInstanceRemoved: (state, action: PayloadAction<{ canvasId: string }>) => {
      delete state.instances[action.payload.canvasId];
    },
    activeCanvasChanged: (state, action: PayloadAction<{ canvasId: string | null }>) => {
      state.activeInstanceId = action.payload.canvasId;
    },
  },
  extraReducers: (builder) => {
    builder.addMatcher(
      isAnyOf(...Object.values(instanceActions)),
      (state, action) => {
        const canvasId = (action as PayloadAction).payload?.canvasId;
        if (canvasId && state.instances[canvasId]) {
          state.instances[canvasId] = undoableCanvasInstanceReducer(state.instances[canvasId], action);
        }
      }
    );
  },
});
```

### 1.3 Update Selectors

**File**: `src/features/controlLayers/store/selectors.ts`

Selectors must be updated to account for the `Undoable` state shape, accessing the `.present` property for the current state.

```typescript
// Old
export const selectCanvasSlice = (state: RootState) => state.canvas;

// New
export const selectCanvasInstance = (state: RootState, canvasId: string) => 
  state.canvases.instances[canvasId]?.present;

export const selectActiveCanvas = (state: RootState) => 
  state.canvases.instances[state.canvases.activeInstanceId]?.present;
```

### 1.4 Migration Strategy

**File**: `src/app/store/migrations/canvasMigration.ts`

The migration script needs to wrap the old canvas state in the new `instances` and `Undoable` structures.

```typescript
const migrateCanvasV1ToV2 = (state: any) => {
  if (state.canvas && !state.canvases) {
    const canvasId = nanoid();
    const undoableState = {
      past: [],
      present: state.canvas, // The old state becomes the 'present'
      future: [],
    };
    return {
      ...state,
      canvases: {
        instances: {
          [canvasId]: undoableState
        },
        activeInstanceId: canvasId
      }
    };
  }
  return state;
};
```

## Phase 2: Canvas Manager Factory Pattern

### 2.0 Configure Listener Middleware

**File**: `src/app/store/store.ts`

To enable efficient, targeted state subscriptions, the listener middleware must be added to the store. This is a one-time setup.

```typescript
import { configureStore, createListenerMiddleware } from '@reduxjs/toolkit';

// Create and export the middleware instance
export const listenerMiddleware = createListenerMiddleware();

export const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) =>
    // Prepend the listener middleware to the chain
    getDefaultMiddleware().prepend(listenerMiddleware.middleware),
});
```

### 2.1 Replace Singleton with Registry

**File**: `src/features/controlLayers/store/ephemeral.ts`

```typescript
// Old
export const $canvasManager = atom<CanvasManager | null>(null);

// New
export const $canvasManagers = atom<Map<string, CanvasManager>>(new Map());
```

### 2.2 Update Canvas Manager Factory

**File**: `src/features/controlLayers/konva/CanvasManagerFactory.ts`

The factory will now manage the lifecycle of state listeners, creating and destroying them alongside the canvas manager instances.

```typescript
import { listenerMiddleware } from 'src/app/store/store';
import { selectCanvasInstance } from 'src/features/controlLayers/store/selectors';

export class CanvasManagerFactory {
  private managers = new Map<string, CanvasManager>();
  private unsubscribers = new Map<string, () => void>();

  createInstance(
    canvasId: string,
    container: HTMLDivElement,
    store: AppStore,
    socket: SocketClient
  ): CanvasManager {
    const manager = new CanvasManager(container, store, socket, canvasId);
    this.managers.set(canvasId, manager);

    const listener = listenerMiddleware.startListening({
      predicate: (action, currentState, previousState) => {
        const oldState = selectCanvasInstance(previousState, canvasId);
        const newState = selectCanvasInstance(currentState, canvasId);
        return oldState !== newState;
      },
      effect: (action, listenerApi) => {
        const latestState = selectCanvasInstance(listenerApi.getState(), canvasId);
        if (latestState) {
          manager.onStateUpdated(latestState);
        }
      },
    });

    this.unsubscribers.set(canvasId, listener.unsubscribe);
    return manager;
  }

  getInstance(canvasId: string): CanvasManager | undefined {
    return this.managers.get(canvasId);
  }

  destroyInstance(canvasId: string): void {
    this.unsubscribers.get(canvasId)?.();
    this.unsubscribers.delete(canvasId);

    const manager = this.managers.get(canvasId);
    if (manager) {
      manager.destroy();
      this.managers.delete(canvasId);
    }
  }
}
```

### 2.3 Update Canvas Manager

**File**: `src/features/controlLayers/konva/CanvasManager.ts`

The manager is now simplified. It no longer subscribes to the store directly, but instead has a new method, `onStateUpdated`, which is called by the listener middleware.

```typescript
// The constructor no longer manages its own subscription.
constructor(
  container: HTMLDivElement,
  store: AppStore,
  socket: SocketClient,
  private canvasId: string
) {
  // Initial state can be read once.
  const initialState = selectCanvasInstance(store.getState(), this.canvasId);
  if (initialState) {
    this.onStateUpdated(initialState);
  }
}

// New method to be called by the listener middleware's effect.
public onStateUpdated(state: CanvasState): void {
  // All logic that used to be in the store.subscribe callback goes here.
}
```

## Phase 3: Dockview Integration

### 3.1 Update Canvas Tab Layout

**File**: `src/features/ui/layouts/canvas-tab-auto-layout.tsx`

Modify main panel initialization to support multiple workspace panels:
```typescript
const initializeCenterPanelLayout = (tab: TabName, api: DockviewApi) => {
  navigationApi.registerContainer(tab, 'main', api, () => {
    // ... existing launchpad and viewer setup
    
    // Create first canvas instance
    const firstCanvasId = nanoid();
    api.addPanel<DockviewPanelParameters>({
      id: `${WORKSPACE_PANEL_ID}_${firstCanvasId}`,
      component: WORKSPACE_PANEL_ID,
      title: 'Canvas 1',
      tabComponent: DOCKVIEW_TAB_CANVAS_WORKSPACE_ID,
      params: {
        tab,
        canvasId: firstCanvasId,
        focusRegion: 'canvas',
        i18nKey: 'ui.panels.canvas',
      },
      position: {
        direction: 'within',
        referencePanel: launchpad.id,
      },
    });
  });
};
```

### 3.2 Add Canvas Management Actions

**File**: `src/features/ui/components/CanvasInstanceManager.tsx`

```typescript
export const CanvasInstanceManager = () => {
  const dispatch = useAppDispatch();
  const canvasCount = useAppSelector(selectCanvasCount);
  
  const addCanvas = useCallback(() => {
    if (canvasCount >= 3) return;
    
    const canvasId = nanoid();
    const canvasName = `Canvas ${canvasCount + 1}`;
    
    // Add to Redux
    dispatch(canvasInstanceAdded({ canvasId, name: canvasName }));
    
    // Add to dockview
    const api = getDockviewApi('canvas', 'main');
    api?.addPanel({
      id: `${WORKSPACE_PANEL_ID}_${canvasId}`,
      component: WORKSPACE_PANEL_ID,
      title: canvasName,
      params: { canvasId },
    });
  }, [canvasCount, dispatch]);
  
  // ... close canvas handler
};
```

## Phase 4: Context-Based Canvas API

### 4.1 Canvas Instance Context

**File**: `src/features/controlLayers/contexts/CanvasInstanceContext.tsx`

```typescript
interface CanvasInstanceContextValue {
  canvasId: string;
  canvasName: string;
  manager: CanvasManager;
  dispatch: (action: CanvasAction) => void;
  useSelector: <T>(selector: (state: CanvasState) => T) => T;
}

export const CanvasInstanceProvider: React.FC<{
  canvasId: string;
  children: React.ReactNode;
}> = ({ canvasId, children }) => {
  const store = useAppStore();
  const manager = useCanvasManager(canvasId);
  
  const dispatch = useCallback((action: CanvasAction) => {
    store.dispatch({ ...action, canvasId });
  }, [store, canvasId]);
  
  const useSelector = useCallback(<T,>(selector: (state: CanvasState) => T) => {
    return useAppSelector((state) => 
      selector(selectCanvasInstance(state, canvasId))
    );
  }, [canvasId]);
  
  const value = useMemo(() => ({
    canvasId,
    manager,
    dispatch,
    useSelector,
  }), [canvasId, manager, dispatch, useSelector]);
  
  return (
    <CanvasInstanceContext.Provider value={value}>
      {children}
    </CanvasInstanceContext.Provider>
  );
};
```

### 4.2 Update CanvasWorkspacePanel

**File**: `src/features/ui/layouts/CanvasWorkspacePanel.tsx`

```typescript
export const CanvasWorkspacePanel = memo(({ params }: DockviewPanelProps) => {
  const { canvasId } = params as { canvasId: string };
  
  return (
    <CanvasInstanceProvider canvasId={canvasId}>
      <StagingAreaContextProvider sessionId={sessionId}>
        {/* ... existing content but now scoped to canvasId */}
      </StagingAreaContextProvider>
    </CanvasInstanceProvider>
  );
});
```

## Phase 5: Generation Pipeline Updates

### 5.1 Update Readiness Checks

**File**: `src/features/queue/store/readiness.ts`

```typescript
const getReasonsWhyCannotEnqueueCanvasTab = (arg: {
  activeCanvasId: string | null;
  canvasManagers: Map<string, CanvasManager>;
  // ... other args
}) => {
  if (!activeCanvasId) {
    return [{ content: 'No active canvas' }];
  }
  
  const canvas = canvases[activeCanvasId];
  const manager = canvasManagers.get(activeCanvasId);
  
  if (!canvas || !manager) {
    return [{ content: 'Canvas not initialized' }];
  }
  
  // ... existing readiness checks using specific canvas/manager
};
```

### 5.2 Update Enqueue Hook

**File**: `src/features/queue/hooks/useEnqueueCanvas.ts`

```typescript
export const useEnqueueCanvas = () => {
  const store = useAppStore();
  const activeCanvasId = useAppSelector(selectActiveCanvasId);
  const canvasManager = useCanvasManager(activeCanvasId);
  
  const enqueue = useCallback((prepend: boolean) => {
    if (!canvasManager || !activeCanvasId) {
      log.error('No active canvas');
      return;
    }
    
    return enqueueCanvas(store, canvasManager, activeCanvasId, prepend);
  }, [canvasManager, activeCanvasId, store]);
  
  return enqueue;
};
```

## Phase 6: Component Migration

### 6.1 Update Canvas Hooks and Action Dispatching

All canvas-related hooks and action dispatching need to use the new context-aware and instance-aware patterns.

**Drawing Actions**: Use the `dispatch` function from the `useCanvasContext` hook, which automatically injects the correct `canvasId`.

```typescript
// Old
export const useAddRasterLayer = () => {
  const dispatch = useAppDispatch();
  // ...
  dispatch(rasterLayerAdded(...));
}

// New
import { instanceActions } from '../store/canvasInstanceSlice';

export const useAddRasterLayer = () => {
  const { dispatch, useSelector } = useCanvasContext();
  // ...
  dispatch(instanceActions.rasterLayerAdded(...));
}
```

**Undo/Redo Actions**: To undo/redo the *active* canvas, global hooks will dispatch the standard `redux-undo` actions, but they must be wrapped in a payload that includes the active `canvasId`. The `canvasesSlice` router will intercept this and apply the action to the correct history.

```typescript
import { ActionCreators as UndoActionCreators } from 'redux-undo';

const useUndo = () => {
  const dispatch = useAppDispatch();
  const activeCanvasId = useAppSelector(selectActiveCanvasId);

  return () => {
    if (!activeCanvasId) return;
    dispatch({ ...UndoActionCreators.undo(), payload: { canvasId: activeCanvasId } });
  }
}
```

### 6.2 Component Updates

Update approximately 45 components that use `useCanvasManager` or canvas selectors:
- Replace `useAppDispatch()` with `useCanvasContext().dispatch`
- Replace `useAppSelector(selectCanvas...)` with `useCanvasContext().useSelector(...)`
- Replace `useCanvasManager()` with `useCanvasContext().manager`

## Phase 7: Navigation & Active Canvas Tracking

### 7.1 Track Active Canvas in Dockview

**File**: `src/features/ui/layouts/canvas-tab-auto-layout.tsx`

```typescript
// Track active workspace panel
api.onDidActivePanelChange((panel) => {
  if (panel?.id.startsWith(WORKSPACE_PANEL_ID)) {
    const canvasId = panel.params?.canvasId;
    // When a canvas panel is activated, its canvasId is set as the active canvas ID in redux.
    // When a non-canvas panel is activated, the active canvas ID is set to null.
    dispatch(setActiveCanvasId(canvasId ?? null));
  }
});
```

### 7.2 Canvas Tab Management UI

Add UI controls for:
- Add new canvas button (max 3)
- Close canvas (with confirmation if has changes)
- Rename canvas
- Canvas count indicator

## Testing Strategy

### Unit Tests
- Redux slice with multiple instances
- Selector parameterization
- Context provider isolation
- Canvas manager lifecycle

### Integration Tests
- Canvas creation/deletion
- Switching between canvases
- Generation from correct canvas
- State isolation between instances

### E2E Tests
- Full workflow with multiple canvases
- Memory cleanup on canvas close
- Persistence and restoration

## Migration Considerations

### Backward Compatibility
- Migrate existing single canvas to first instance
- Preserve all canvas state during migration
- Version state structure for future migrations

### Performance Monitoring
- Track memory usage with multiple canvases
- Monitor React re-render patterns
- Profile Canvas manager instances

### Rollout Strategy
1. Feature flag for multi-instance UI
2. Beta testing with power users
3. Gradual rollout with monitoring
4. Full release after stability confirmation

## File Structure Changes

```
src/features/controlLayers/
├── store/
│   ├── canvasesSlice.ts (renamed from canvasSlice.ts)
│   ├── canvasSelectors.ts (parameterized selectors)
│   └── canvasMigrations.ts (new)
├── contexts/
│   ├── CanvasInstanceContext.tsx (new)
│   └── CanvasManagerProviderGate.tsx (updated)
├── konva/
│   ├── CanvasManager.ts (add canvasId support)
│   └── CanvasManagerFactory.ts (new)
└── hooks/
    └── useCanvasContext.ts (new)
```

## Key Technical Decisions

1. **Use Dockview's native tabs** rather than implementing custom tab UI
2. **Context-based API abstraction** to hide multi-instance complexity from components
3. **Active canvas tracking in Redux** for generation pipeline
4. **Manager registry pattern** for lifecycle management
5. **Parameterized selectors** rather than duplicating selector code
6. **Migration-first approach** to preserve existing user work

## Risk Mitigation

- **Memory leaks**: Implement proper cleanup in manager lifecycle
- **State corruption**: Add validation and error boundaries per canvas
- **Performance degradation**: Lazy load non-active canvases
- **User confusion**: Clear active canvas indication in UI
- **Data loss**: Auto-save and recovery mechanisms

## Success Metrics

- Support 3 simultaneous canvas instances without performance degradation
- Zero data loss during canvas operations
- Maintain current generation speed
- Clean component API with minimal changes
- Successful migration of existing canvases