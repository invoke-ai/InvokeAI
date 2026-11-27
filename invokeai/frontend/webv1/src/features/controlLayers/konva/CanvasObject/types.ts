import type { CanvasObjectBrushLine } from 'features/controlLayers/konva/CanvasObject/CanvasObjectBrushLine';
import type { CanvasObjectBrushLineWithPressure } from 'features/controlLayers/konva/CanvasObject/CanvasObjectBrushLineWithPressure';
import type { CanvasObjectEraserLine } from 'features/controlLayers/konva/CanvasObject/CanvasObjectEraserLine';
import type { CanvasObjectEraserLineWithPressure } from 'features/controlLayers/konva/CanvasObject/CanvasObjectEraserLineWithPressure';
import type { CanvasObjectGradient } from 'features/controlLayers/konva/CanvasObject/CanvasObjectGradient';
import type { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import type { CanvasObjectLasso } from 'features/controlLayers/konva/CanvasObject/CanvasObjectLasso';
import type { CanvasObjectOval } from 'features/controlLayers/konva/CanvasObject/CanvasObjectOval';
import type { CanvasObjectPolygon } from 'features/controlLayers/konva/CanvasObject/CanvasObjectPolygon';
import type { CanvasObjectRect } from 'features/controlLayers/konva/CanvasObject/CanvasObjectRect';
import type {
  CanvasBrushLineState,
  CanvasBrushLineWithPressureState,
  CanvasEraserLineState,
  CanvasEraserLineWithPressureState,
  CanvasGradientState,
  CanvasImageState,
  CanvasLassoState,
  CanvasOvalState,
  CanvasPolygonState,
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
  | CanvasObjectLasso
  | CanvasObjectOval
  | CanvasObjectPolygon
  | CanvasObjectImage
  | CanvasObjectGradient;
/**
 * Union of all object states.
 */
export type AnyObjectState =
  | CanvasBrushLineState
  | CanvasBrushLineWithPressureState
  | CanvasEraserLineState
  | CanvasEraserLineWithPressureState
  | CanvasImageState
  | CanvasRectState
  | CanvasLassoState
  | CanvasOvalState
  | CanvasPolygonState
  | CanvasGradientState;
