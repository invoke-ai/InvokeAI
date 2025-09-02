# Phase 6: Component Migration Work Log

## Overview
Phase 6 involves migrating approximately 45 components to use the new context-based API and instance-aware patterns. This is the largest phase of the Canvas Multi-Instance Implementation Plan.

## Key Tasks
1. **Section 6.1**: Update Canvas Hooks and Action Dispatching
   - Replace useAppDispatch with useCanvasContext().dispatch for canvas actions
   - Update action creators to use instanceActions from canvasInstanceSlice
   - Implement undo/redo with canvasId routing

2. **Section 6.2**: Component Updates
   - Update components that use canvas selectors to use context-based selectors
   - Replace direct useCanvasManager() with useCanvasContext().manager
   - Update all imports to use new action creators

## Progress Log

### Phase 6.0: Setup and Analysis
- [x] Create phase6_worklog.md
- [x] Examine current state of canvasInstanceSlice and related files
- [x] Find components that import from canvasSlice (now canvasesSlice)
- [x] Identify hooks that dispatch canvas actions
- [x] Create systematic update plan

## Analysis Results
Found **69 files** using `useAppDispatch` that need context-based dispatch
Found **48 files** importing from `canvasSlice` that need to use `instanceActions`

### Infrastructure Status (Phases 1-5 Complete)
- ✅ `canvasInstanceSlice.ts` with all `instanceActions`
- ✅ `canvasesSlice.ts` as router slice with undo/redo
- ✅ `CanvasInstanceContext.tsx` providing context-based API
- ✅ Registry pattern with `$canvasManagers` atom

### Phase 6.1: Update Canvas Hooks and Action Dispatching
- [ ] Update undo/redo hooks to use canvas-aware actions
- [ ] Update hooks that dispatch canvas drawing actions
- [ ] Update action creators to use instanceActions

### Phase 6.2: Component Updates
- [ ] Update components using useAppDispatch → useCanvasContext().dispatch
- [ ] Update components importing from canvasSlice → instanceActions
- [ ] Replace useCanvasManager() → useCanvasContext().manager
- [ ] Update selectors to use context-based selectors

## Files to Update by Category

### Priority 1: Undo/Redo & Core Hooks (3 files)
1. `hooks/useCanvasUndoRedoHotkeys.tsx` - Global undo/redo
2. `components/Toolbar/CanvasToolbarUndoButton.tsx`
3. `components/Toolbar/CanvasToolbarRedoButton.tsx`

### Priority 2: Canvas Drawing Actions (8 files)  
1. `hooks/addLayerHooks.ts`
2. `hooks/saveCanvasHooks.ts`
3. `hooks/useInvertMask.ts`
4. Canvas entity manipulation hooks

### Priority 3: Component Actions (~58 files)
All components using useAppDispatch for canvas state updates

## Progress Update - Phase 6.1 and 6.2 Core Hooks Complete ✅
**Phase 6.1 Completed:**
- [x] ✅ Updated undo/redo hotkeys hook  
- [x] ✅ Updated undo/redo toolbar buttons
- [x] ✅ Updated clear history button

**Phase 6.2 Core Hooks Completed:**
- [x] ✅ Updated `addLayerHooks.ts` - 10 hooks using context dispatch and instanceActions
- [x] ✅ Updated `saveCanvasHooks.ts` - 9 hooks using context dispatch and instanceActions  
- [x] ✅ Updated `useInvertMask.ts` - Mask inversion with context dispatch
- [x] ✅ Fixed selector compatibility issues (context useSelector expects CanvasState, not RootState)
- [x] ✅ Created atomic commits for each meaningful change

**Current Status:**
- ✅ All critical canvas drawing hooks updated
- ✅ TypeScript errors resolved for updated hooks
- ⚠️ ~60 components still need context migration
- ⚠️ Null safety issues remain for components using legacy selectors

**Key Learning:** 
Context `useSelector` gets `CanvasState` directly, not `RootState`. 
Use `useSelector((state) => state.selectedEntityIdentifier)` instead of `useSelector(selectSelectedEntityIdentifier)`.

## Strategy for Remaining Component Updates:

### Systematic Approach for 60+ Remaining Components:

**Step 1: Identify Component Categories**
1. **High-Priority:** Components with canvas drawing actions (entity manipulation, layer operations)
2. **Medium-Priority:** Components that read canvas state (display, validation, UI state)
3. **Low-Priority:** Components with settings/config changes (non-critical for core functionality)

**Step 2: Update Patterns**

For components inside canvas context (children of CanvasWorkspacePanel):
```typescript
// Old
const dispatch = useAppDispatch();
const canvas = useAppSelector(selectCanvasSlice);
dispatch(rasterLayerAdded(...));

// New  
const { dispatch, useSelector } = useCanvasContext();
const canvas = useSelector((state) => state);  // Direct canvas state access
dispatch(instanceActions.rasterLayerAdded(...));
```

For components outside canvas context (global UI, hotkeys):
```typescript
// Keep useAppDispatch but use router actions for active canvas
const dispatch = useAppDispatch();
dispatch(canvasUndo({})); // Routes to active canvas
```

**Step 3: Selector Updates**
- Replace `useAppSelector(selectCanvasSlice)` with context `useSelector((state) => state)`
- Replace `useAppSelector(selectSelectedEntityIdentifier)` with `useSelector((state) => state.selectedEntityIdentifier)`
- Add null checks for components that might not have canvas context

**Step 4: Action Updates**  
- Import `instanceActions` from `canvasInstanceSlice` instead of actions from `canvasSlice`
- Use `dispatch(instanceActions.actionName(...))` within context
- Actions automatically get `canvasId` injected by context dispatch

## Notes
- Focus on components inside the canvas context (children of CanvasWorkspacePanel)
- Components that dispatch canvas drawing actions are highest priority
- Components that access canvas state need context-based selectors
- Phases 1-5 have been completed, so canvasInstanceSlice and context should exist