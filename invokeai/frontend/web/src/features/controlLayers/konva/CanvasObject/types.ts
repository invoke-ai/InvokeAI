import type { CanvasObjectBrushLine } from 'features/controlLayers/konva/CanvasObject/CanvasObjectBrushLine';
import type { CanvasObjectEraserLine } from 'features/controlLayers/konva/CanvasObject/CanvasObjectEraserLine';
import type { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import type { CanvasObjectRect } from 'features/controlLayers/konva/CanvasObject/CanvasObjectRect';
import type {
  CanvasBrushLineState,
  CanvasEraserLineState,
  CanvasImageState,
  CanvasRectState,
} from 'features/controlLayers/store/types';

/**
 * Union of all object renderers.
 */

export type AnyObjectRenderer =
  | CanvasObjectBrushLine
  | CanvasObjectEraserLine
  | CanvasObjectRect
  | CanvasObjectImage; /**
 * Union of all object states.
 */
export type AnyObjectState = CanvasBrushLineState | CanvasEraserLineState | CanvasImageState | CanvasRectState;
