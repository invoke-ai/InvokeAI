import type { CanvasObjectBrushLine } from 'features/controlLayers/konva/CanvasObject/CanvasObjectBrushLine';
import type { CanvasObjectBrushLineWithPressure } from 'features/controlLayers/konva/CanvasObject/CanvasObjectBrushLineWithPressure';
import type { CanvasObjectEraserLine } from 'features/controlLayers/konva/CanvasObject/CanvasObjectEraserLine';
import type { CanvasObjectEraserLineWithPressure } from 'features/controlLayers/konva/CanvasObject/CanvasObjectEraserLineWithPressure';
import type { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import type { CanvasObjectRect } from 'features/controlLayers/konva/CanvasObject/CanvasObjectRect';
import type {
  CanvasBrushLineState,
  CanvasBrushLineWithPressureState,
  CanvasEraserLineState,
  CanvasEraserLineWithPressureState,
  CanvasImageState,
  CanvasRectState,
} from 'features/controlLayers/store/types';

/**
 * Union of all object renderers.
 */

export type AnyObjectRenderer =
  | CanvasObjectBrushLine
  | CanvasObjectBrushLineWithPressure
  | CanvasObjectEraserLine
  | CanvasObjectEraserLineWithPressure
  | CanvasObjectRect
  | CanvasObjectImage; /**
 * Union of all object states.
 */
export type AnyObjectState =
  | CanvasBrushLineState
  | CanvasBrushLineWithPressureState
  | CanvasEraserLineState
  | CanvasEraserLineWithPressureState
  | CanvasImageState
  | CanvasRectState;
