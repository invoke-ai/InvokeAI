# Phase 2: Canvas Manager Factory Pattern - Work Log

## Overview
Implementing Phase 2 of the Canvas Multi-Instance Implementation Plan:
- Section 2.0: Configure Listener Middleware in the store
- Section 2.1: Replace singleton $canvasManager with registry Map
- Section 2.2: Update/Create Canvas Manager Factory with lifecycle management
- Section 2.3: Update Canvas Manager to work with the new architecture

## Progress Log

### Started: 2025-09-02

#### Initial Assessment
- [x] Examine current store configuration
  - Listener middleware is already configured in store.ts (lines 59, 201)
  - canvasesSlice is already implemented with multi-instance support
- [x] Review existing CanvasManager implementation 
  - Located at `/src/features/controlLayers/konva/CanvasManager.ts`
  - Uses singleton pattern via $canvasManager atom
- [x] Check current singleton pattern usage
  - Current singleton: `$canvasManager` atom in ephemeral.ts
  - Need to replace with Map-based registry

#### Section 2.0: Configure Listener Middleware
- [x] Add listener middleware to store
  - Already configured in store.ts (lines 59, 201)
- [x] Configure listeners for canvas instance management
  - listenerMiddleware is exported and ready for use

#### Section 2.1: Replace Singleton with Registry
- [x] Replace $canvasManager singleton with Map-based registry
  - Updated ephemeral.ts to use $canvasManagers Map
- [ ] Update all imports and references (deferred to Section 2.3)

#### Section 2.2: Canvas Manager Factory
- [x] Create/Update factory pattern for canvas managers
  - Created CanvasManagerFactory.ts with full lifecycle management
- [x] Implement lifecycle management (creation, cleanup)
  - Includes state listener setup/teardown
  - Registry management with proper cleanup
  - Singleton factory instance exported

#### Section 2.3: Update Canvas Manager Architecture
- [ ] Update CanvasManager to work with new architecture
- [ ] Ensure proper instance isolation

## Notes
- Working directory: /home/bat/git/InvokeAI/invokeai/frontend/web
- Current branch: psyche/feat/ui/multi-instance-canvas
- Phase 1 completed: canvasInstanceSlice and canvasesSlice are in place