# Hotkeys System

This document describes the technical implementation of the customizable hotkeys system in InvokeAI.

> **Note:** For user-facing documentation on how to use customizable hotkeys, see [Hotkeys Feature Documentation](../features/hotkeys.md).

## Overview

The hotkeys system allows users to customize keyboard shortcuts throughout the application. All hotkeys are:
- Centrally defined and managed
- Customizable by users
- Persisted across sessions
- Type-safe and validated

## Architecture

The customizable hotkeys feature is built on top of the existing hotkey system with the following components:

### 1. Hotkeys State Slice (`hotkeysSlice.ts`)

Location: `invokeai/frontend/web/src/features/system/store/hotkeysSlice.ts`

**Responsibilities:**
- Stores custom hotkey mappings in Redux state
- Persisted to IndexedDB using `redux-remember`
- Provides actions to change, reset individual, or reset all hotkeys

**State Shape:**
```typescript
{
  _version: 1,
  customHotkeys: {
    'app.invoke': ['mod+enter'],
    'canvas.undo': ['mod+z'],
    // ...
  }
}
```

**Actions:**
- `hotkeyChanged(id, hotkeys)` - Update a single hotkey
- `hotkeyReset(id)` - Reset a single hotkey to default
- `allHotkeysReset()` - Reset all hotkeys to defaults

### 2. useHotkeyData Hook (`useHotkeyData.ts`)

Location: `invokeai/frontend/web/src/features/system/components/HotkeysModal/useHotkeyData.ts`

**Responsibilities:**
- Defines all default hotkeys
- Merges default hotkeys with custom hotkeys from the store
- Returns the effective hotkeys that should be used throughout the app
- Provides platform-specific key translations (Ctrl/Cmd, Alt/Option)

**Key Functions:**
- `useHotkeyData()` - Returns all hotkeys organized by category
- `useRegisteredHotkeys()` - Hook to register a hotkey in a component

### 3. HotkeyEditor Component (`HotkeyEditor.tsx`)

Location: `invokeai/frontend/web/src/features/system/components/HotkeysModal/HotkeyEditor.tsx`

**Features:**
- Inline editor with input field
- Modifier buttons (Mod, Ctrl, Shift, Alt) for quick insertion
- Live preview of hotkey combinations
- Validation with visual feedback
- Help tooltip with syntax examples
- Save/cancel/reset buttons

**Smart Features:**
- Automatic `+` insertion between modifiers
- Cursor position preservation
- Validation prevents invalid combinations (e.g., modifier-only keys)

### 4. HotkeysModal Component (`HotkeysModal.tsx`)

Location: `invokeai/frontend/web/src/features/system/components/HotkeysModal/HotkeysModal.tsx`

**Features:**
- View Mode / Edit Mode toggle
- Search functionality
- Category-based organization
- Shows HotkeyEditor components when in edit mode
- "Reset All to Default" button in edit mode

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User opens Hotkeys Modal                                 │
│ 2. User clicks "Edit Mode" button                           │
│ 3. User clicks edit icon next to a hotkey                   │
│ 4. User enters new hotkey(s) using editor                   │
│ 5. User clicks save or presses Enter                        │
│ 6. Custom hotkey stored via hotkeyChanged() action          │
│ 7. Redux state persisted to IndexedDB (redux-remember)      │
│ 8. useHotkeyData() hook picks up the change                 │
│ 9. All components using useRegisteredHotkeys() get update   │
└─────────────────────────────────────────────────────────────┘
```

## Hotkey Format

Hotkeys use the format from `react-hotkeys-hook` library:

- **Modifiers:** `mod`, `ctrl`, `shift`, `alt`, `meta`
- **Keys:** Letters, numbers, function keys, special keys
- **Separator:** `+` between keys in a combination
- **Multiple hotkeys:** Comma-separated (e.g., `mod+a, ctrl+b`)

**Examples:**
- `mod+enter` - Mod key + Enter
- `shift+x` - Shift + X
- `ctrl+shift+a` - Control + Shift + A
- `f1, f2` - F1 or F2 (alternatives)

## Developer Guide

### Using Hotkeys in Components

To use a hotkey in a component:

```tsx
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';

const MyComponent = () => {
  const handleAction = useCallback(() => {
    // Your action here
  }, []);

  // This automatically uses custom hotkeys if configured
  useRegisteredHotkeys({
    id: 'myAction',
    category: 'app', // or 'canvas', 'viewer', 'gallery', 'workflows'
    callback: handleAction,
    options: { enabled: true, preventDefault: true },
    dependencies: [handleAction]
  });

  // ...
};
```

**Options:**
- `enabled` - Whether the hotkey is active
- `preventDefault` - Prevent default browser behavior
- `enableOnFormTags` - Allow hotkey in form elements (default: false)

### Adding New Hotkeys

To add a new hotkey to the system:

#### 1. Add Translation Strings

In `invokeai/frontend/web/public/locales/en.json`:

```json
{
  "hotkeys": {
    "app": {
      "myAction": {
        "title": "My Action",
        "desc": "Description of what this hotkey does"
      }
    }
  }
}
```

#### 2. Register the Hotkey

In `invokeai/frontend/web/src/features/system/components/HotkeysModal/useHotkeyData.ts`:

```typescript
// Inside the appropriate category builder function
addHotkey('app', 'myAction', ['mod+k']); // Default binding
```

#### 3. Use the Hotkey

In your component:

```typescript
useRegisteredHotkeys({
  id: 'myAction',
  category: 'app',
  callback: handleMyAction,
  options: { enabled: true },
  dependencies: [handleMyAction]
});
```

### Hotkey Categories

Current categories:
- **app** - Global application hotkeys
- **canvas** - Canvas/drawing operations
- **viewer** - Image viewer operations
- **gallery** - Gallery/image grid operations
- **workflows** - Node workflow editor

To add a new category, update `useHotkeyData.ts` and add translations.

## Testing

Tests are located in `invokeai/frontend/web/src/features/system/store/hotkeysSlice.test.ts`.

**Test Coverage:**
- Adding custom hotkeys
- Updating existing custom hotkeys
- Resetting individual hotkeys
- Resetting all hotkeys
- State persistence and migration

Run tests with:

```bash
cd invokeai/frontend/web
pnpm test:no-watch
```

## Persistence

Custom hotkeys are persisted using the same mechanism as other app settings:

- Stored in Redux state under the `hotkeys` slice
- Persisted to IndexedDB via `redux-remember`
- Automatically loaded when the app starts
- Survives page refreshes and browser restarts
- Includes migration support for state schema changes

**State Location:**
- IndexedDB database: `invoke`
- Store key: `hotkeys`

## Dependencies

- **react-hotkeys-hook** (v4.5.0) - Core hotkey handling
- **@reduxjs/toolkit** - State management
- **redux-remember** - Persistence
- **zod** - State validation

## Best Practices

1. **Use `mod` instead of `ctrl`** - Automatically maps to Cmd on Mac, Ctrl elsewhere
2. **Provide descriptive translations** - Help users understand what each hotkey does
3. **Avoid conflicts** - Check existing hotkeys before adding new ones
4. **Use preventDefault** - Prevent browser default behavior when appropriate
5. **Check enabled state** - Only activate hotkeys when the action is available
6. **Use dependencies correctly** - Ensure callbacks are stable with useCallback

## Common Patterns

### Conditional Hotkeys

```typescript
useRegisteredHotkeys({
  id: 'save',
  category: 'app',
  callback: handleSave,
  options: {
    enabled: hasUnsavedChanges && !isLoading, // Only when valid
    preventDefault: true
  },
  dependencies: [hasUnsavedChanges, isLoading, handleSave]
});
```

### Multiple Hotkeys for Same Action

```typescript
// In useHotkeyData.ts
addHotkey('canvas', 'redo', ['mod+shift+z', 'mod+y']); // Two alternatives
```

### Focus-Scoped Hotkeys

```typescript
import { useFocusRegion } from 'common/hooks/focus';

const MyComponent = () => {
  const focusRegionRef = useFocusRegion('myRegion');

  // Hotkey only works when this region has focus
  useRegisteredHotkeys({
    id: 'myAction',
    category: 'app',
    callback: handleAction,
    options: { enabled: true }
  });

  return <div ref={focusRegionRef}>...</div>;
};
```
