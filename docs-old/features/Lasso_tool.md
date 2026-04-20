Lasso Tool
===========

- The Lasso tool creates selections and inpaint masks by drawing freehand or polygonal regions on the canvas.

How to open the Lasso tool
--------------------------
- Click the Lasso icon in the toolbar.
- Hotkey: press `L` (default). The hotkey is shown in the tool's tooltip and can be customized in Hotkeys settings.

Modes
-----
- Freehand (default)
  - Hold the pointer and drag to draw a continuous contour.
  - Long segments are broken into intermediate points to keep the line continuous.
  - Very long strokes may be simplified after drawing to reduce point count for performance.

- Polygon
  - Click to place points; click the first point (or a point near it) to close the polygon.
  - The tool snaps the closing point to the start for precise closures.

Basic interactions
------------------
- Switch modes with the mode toggle in the toolbar.
- To close a polygon: click the starting point again or click near it — the tool aligns the final point to the start to complete the shape.
- The selection will be added to the current Inpaint Mask layer. If no Inpaint Mask layer exists, a new one will be created automatically.

Tips & behavior
---------------
- Hold `Space` to temporarily switch to the View tool for panning and zooming; release `Space` to return to the Lasso tool and continue drawing.
- When using the Polygon mode, you can hold `Shift` to snap points to horizontal, vertical, or 45-degree angles for more precise shapes.
- Hold `Ctrl` (Windows/Linux) or `Command` (macOS) while drawing to subtract from the current selection instead of adding to it.
