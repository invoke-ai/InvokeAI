# Text Tool

## Font selection

The Text tool uses a set of predefined font stacks. When you choose a font, the app resolves the first available font on your system from that stack and uses it for both the editor overlay and the rasterized result. This provides consistent styling across platforms while still falling back to safe system fonts if a preferred font is missing.

## Size and spacing

- **Size** controls the font size in pixels.
- **Spacing** controls the line height multiplier (Dense, Normal, Spacious). This affects the distance between lines while editing the text.

## Uncommitted state

While text is uncommitted, it remains editable on-canvas. Access to other tools is blocked. Switching to other tabs (Generate, Upascaling, Workflows etc.) discards the text. The uncommitted box can be moved and rotated:

- **Move:** Hold Ctrl (Windows/Linux) or Command (macOS) and drag to move the text box.
- **Rotate:** Drag the rotation handle above the box. Hold **Shift** while rotating to snap to 15 degree increments.

The text is committed to a raster layer when you press **Enter**. Press **Esc** to discard the current text session.
