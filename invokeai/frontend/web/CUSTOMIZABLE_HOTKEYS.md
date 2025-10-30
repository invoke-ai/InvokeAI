# Customizable Hotkeys

This feature allows users to customize keyboard shortcuts (hotkeys) in the InvokeAI frontend application.

## Overview

Users can now:
- View all available hotkeys in the application
- Edit individual hotkeys to their preference
- Reset individual hotkeys to their defaults
- Reset all hotkeys to defaults
- Have their custom hotkeys persist across sessions

## Implementation

### Architecture

The customizable hotkeys feature is built on top of the existing hotkey system with the following components:

1. **Hotkeys State Slice** (`hotkeysSlice.ts`)
   - Stores custom hotkey mappings in Redux state
   - Persisted to IndexedDB using `redux-remember`
   - Provides actions to change, reset individual, or reset all hotkeys

2. **useHotkeyData Hook** (`useHotkeyData.ts`)
   - Updated to merge default hotkeys with custom hotkeys from the store
   - Returns the effective hotkeys that should be used throughout the app

3. **HotkeyEditor Component** (`HotkeyEditor.tsx`)
   - Provides UI for editing individual hotkeys
   - Shows inline editor with save/cancel buttons
   - Displays reset button for customized hotkeys

4. **HotkeysModal Updates** (`HotkeysModal.tsx`)
   - Added "Edit Mode" / "View Mode" toggle
   - Shows HotkeyEditor components when in edit mode
   - Provides "Reset All to Default" button in edit mode

### Data Flow

1. User opens Hotkeys Modal (Shift+?) 
2. User clicks "Edit Mode" button
3. User clicks the edit icon next to any hotkey
4. User enters new hotkey(s) (comma-separated for multiple)
5. User clicks save or presses Enter
6. Custom hotkey is stored in Redux state via `hotkeyChanged` action
7. Redux state is persisted to IndexedDB via `redux-remember`
8. `useHotkeyData` hook picks up the change and returns updated hotkeys
9. All components using `useRegisteredHotkeys` automatically use the new hotkey

### Hotkey Format

- Hotkeys use the format from `react-hotkeys-hook`
- Multiple hotkeys for the same action are separated by commas
- Modifiers: `mod` (ctrl on Windows/Linux, cmd on Mac), `shift`, `alt`
- Examples: `mod+enter`, `shift+x`, `ctrl+shift+a`

## Usage

### For Users

1. Press `Shift+?` to open the Hotkeys Modal
2. Click "Edit Mode" button at the bottom
3. Click the pencil icon next to the hotkey you want to change
4. Enter the new hotkey(s) (e.g., `ctrl+k` or `ctrl+k, cmd+k` for multiple)
5. Press Enter or click the checkmark to save
6. To reset a single hotkey, click the counter-clockwise arrow icon
7. To reset all hotkeys, click "Reset All to Default" button

### For Developers

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
    options: { enabled: true },
  });

  // ...
};
```

To add a new hotkey to the system:

1. Add translation strings in `public/locales/en.json`:
   ```json
   "hotkeys": {
     "app": {
       "myAction": {
         "title": "My Action",
         "desc": "Description of what this hotkey does"
       }
     }
   }
   ```

2. Register the hotkey in `useHotkeyData.ts`:
   ```typescript
   addHotkey('app', 'myAction', ['mod+k']);
   ```

## Testing

Tests are located in `hotkeysSlice.test.ts` and cover:
- Adding custom hotkeys
- Updating existing custom hotkeys  
- Resetting individual hotkeys
- Resetting all hotkeys

Run tests with:
```bash
pnpm run test
```

## Persistence

Custom hotkeys are persisted using the same mechanism as other app settings:
- Stored in Redux state under the `hotkeys` slice
- Persisted to IndexedDB via `redux-remember`
- Automatically loaded when the app starts
- Survives page refreshes and browser restarts
