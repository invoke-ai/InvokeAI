# Customizable Hotkeys

InvokeAI allows you to customize all keyboard shortcuts (hotkeys) to match your workflow preferences.

## Features

- **View All Hotkeys**: See all available keyboard shortcuts in one place
- **Customize Any Hotkey**: Change any shortcut to your preference
- **Multiple Bindings**: Assign multiple key combinations to the same action
- **Smart Validation**: Built-in validation prevents invalid combinations
- **Persistent Settings**: Your custom hotkeys are saved and restored across sessions
- **Easy Reset**: Reset individual hotkeys or all hotkeys back to defaults

## How to Use

### Opening the Hotkeys Modal

Press `Shift+?` or click the keyboard icon in the application to open the Hotkeys Modal.

### Viewing Hotkeys

In **View Mode** (default), you can:
- Browse all available hotkeys organized by category (App, Canvas, Gallery, Workflows, etc.)
- Search for specific hotkeys using the search bar
- See the current key combination for each action

### Customizing Hotkeys

1. Click the **Edit Mode** button at the bottom of the Hotkeys Modal
2. Find the hotkey you want to change
3. Click the **pencil icon** next to it
4. The editor will appear with:
   - **Input field**: Enter your new hotkey combination
   - **Modifier buttons**: Quick-insert Mod, Ctrl, Shift, Alt keys
   - **Help icon** (?): Shows syntax examples and valid keys
   - **Live preview**: See how your hotkey will look

5. Enter your new hotkey using the format:
   - `mod+a` - Mod key + A (Mod = Ctrl on Windows/Linux, Cmd on Mac)
   - `ctrl+shift+k` - Multiple modifiers
   - `f1` - Function keys
   - `mod+enter, ctrl+enter` - Multiple alternatives (separated by comma)

6. Click the **checkmark** or press Enter to save
7. Click the **X** or press Escape to cancel

### Resetting Hotkeys

**Reset a single hotkey:**
- Click the counter-clockwise arrow icon that appears next to customized hotkeys

**Reset all hotkeys:**
- In Edit Mode, click the **Reset All to Default** button at the bottom

### Hotkey Format Reference

**Valid Modifiers:**
- `mod` - Context-aware: Ctrl (Windows/Linux) or Cmd (Mac)
- `ctrl` - Control key
- `shift` - Shift key
- `alt` - Alt key (Option on Mac)

**Valid Keys:**
- Letters: `a-z`
- Numbers: `0-9`
- Function keys: `f1-f12`
- Special keys: `enter`, `space`, `tab`, `backspace`, `delete`, `escape`
- Arrow keys: `up`, `down`, `left`, `right`
- And more...

**Examples:**
- ✅ `mod+s` - Save action
- ✅ `ctrl+shift+p` - Command palette
- ✅ `f5, mod+r` - Two alternatives for refresh
- ❌ `mod+` - Invalid (no key after modifier)
- ❌ `shift+ctrl+` - Invalid (ends with modifier)

## For Developers

For technical implementation details, architecture, and how to add new hotkeys to the system, see the [Hotkeys Developer Documentation](../contributing/HOTKEYS.md).
