import { calculateNewBrushSize } from 'features/canvas/hooks/useCanvasZoom';
import {
  getIsFocused,
  getIsMouseDown,
  getScaledFlooredCursorPosition,
  snapPosToStage,
} from 'features/controlLayers/konva/util';
import {
  type AddBrushLineArg,
  type AddEraserLineArg,
  type AddPointToLineArg,
  type AddRectShapeArg,
  DEFAULT_RGBA_COLOR,
  type Layer,
  type Tool,
} from 'features/controlLayers/store/types';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import type { WritableAtom } from 'nanostores';
import type { RgbaColor } from 'react-colorful';

import { TOOL_PREVIEW_LAYER_ID } from './naming';

type SetStageEventHandlersArg = {
  stage: Konva.Stage;
  $tool: WritableAtom<Tool>;
  $isDrawing: WritableAtom<boolean>;
  $lastMouseDownPos: WritableAtom<Vector2d | null>;
  $lastCursorPos: WritableAtom<Vector2d | null>;
  $lastAddedPoint: WritableAtom<Vector2d | null>;
  $brushColor: WritableAtom<RgbaColor>;
  $brushSize: WritableAtom<number>;
  $brushSpacingPx: WritableAtom<number>;
  $selectedLayer: WritableAtom<Layer | null>;
  $shouldInvertBrushSizeScrollDirection: WritableAtom<boolean>;
  onBrushLineAdded: (arg: AddBrushLineArg) => void;
  onEraserLineAdded: (arg: AddEraserLineArg) => void;
  onPointAddedToLine: (arg: AddPointToLineArg) => void;
  onRectShapeAdded: (arg: AddRectShapeArg) => void;
  onBrushSizeChanged: (size: number) => void;
};

/**
 * Updates the last cursor position atom with the current cursor position, returning the new position or `null` if the
 * cursor is not over the stage.
 * @param stage The konva stage
 * @param $lastCursorPos The last cursor pos as a nanostores atom
 */
const updateLastCursorPos = (stage: Konva.Stage, $lastCursorPos: WritableAtom<Vector2d | null>) => {
  const pos = getScaledFlooredCursorPosition(stage);
  if (!pos) {
    return null;
  }
  $lastCursorPos.set(pos);
  return pos;
};

/**
 * Adds the next point to a line if the cursor has moved far enough from the last point.
 * @param layerId The layer to (maybe) add the point to
 * @param currentPos The current cursor position
 * @param $lastAddedPoint The last added line point as a nanostores atom
 * @param $brushSpacingPx The brush spacing in pixels as a nanostores atom
 * @param onPointAddedToLine The callback to add a point to a line
 */
const maybeAddNextPoint = (
  layerId: string,
  currentPos: Vector2d,
  $lastAddedPoint: WritableAtom<Vector2d | null>,
  $brushSpacingPx: WritableAtom<number>,
  onPointAddedToLine: (arg: AddPointToLineArg) => void
) => {
  // Continue the last line
  const lastAddedPoint = $lastAddedPoint.get();
  if (lastAddedPoint) {
    // Dispatching redux events impacts perf substantially - using brush spacing keeps dispatches to a reasonable number
    if (Math.hypot(lastAddedPoint.x - currentPos.x, lastAddedPoint.y - currentPos.y) < $brushSpacingPx.get()) {
      return null;
    }
  }
  onPointAddedToLine({ layerId, point: [currentPos.x, currentPos.y] });
};

export const setStageEventHandlers = ({
  stage,
  $tool,
  $isDrawing,
  $lastMouseDownPos,
  $lastCursorPos,
  $lastAddedPoint,
  $brushColor,
  $brushSize,
  $brushSpacingPx,
  $selectedLayer,
  $shouldInvertBrushSizeScrollDirection,
  onBrushLineAdded,
  onEraserLineAdded,
  onPointAddedToLine,
  onRectShapeAdded,
  onBrushSizeChanged,
}: SetStageEventHandlersArg): (() => void) => {
  //#region mouseenter
  stage.on('mouseenter', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const tool = $tool.get();
    stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.visible(tool === 'brush' || tool === 'eraser');
  });

  //#region mousedown
  stage.on('mousedown', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const tool = $tool.get();
    const pos = updateLastCursorPos(stage, $lastCursorPos);
    const selectedLayer = $selectedLayer.get();
    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }
    if (tool === 'brush') {
      onBrushLineAdded({
        layerId: selectedLayer.id,
        points: [pos.x, pos.y, pos.x, pos.y],
        color: selectedLayer.type === 'raster_layer' ? $brushColor.get() : DEFAULT_RGBA_COLOR,
      });
      $isDrawing.set(true);
      $lastMouseDownPos.set(pos);
    }

    if (tool === 'eraser') {
      onEraserLineAdded({
        layerId: selectedLayer.id,
        points: [pos.x, pos.y, pos.x, pos.y],
      });
      $isDrawing.set(true);
      $lastMouseDownPos.set(pos);
    }

    if (tool === 'rect') {
      $isDrawing.set(true);
      $lastMouseDownPos.set(snapPosToStage(pos, stage));
    }
  });

  //#region mouseup
  stage.on('mouseup', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const pos = $lastCursorPos.get();
    const selectedLayer = $selectedLayer.get();

    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }
    const tool = $tool.get();

    if (tool === 'rect') {
      const lastMouseDownPos = $lastMouseDownPos.get();
      if (lastMouseDownPos) {
        const snappedPos = snapPosToStage(pos, stage);
        onRectShapeAdded({
          layerId: selectedLayer.id,
          rect: {
            x: Math.min(snappedPos.x, lastMouseDownPos.x),
            y: Math.min(snappedPos.y, lastMouseDownPos.y),
            width: Math.abs(snappedPos.x - lastMouseDownPos.x),
            height: Math.abs(snappedPos.y - lastMouseDownPos.y),
          },
          color: selectedLayer.type === 'raster_layer' ? $brushColor.get() : DEFAULT_RGBA_COLOR,
        });
      }
    }

    $isDrawing.set(false);
    $lastMouseDownPos.set(null);
  });

  //#region mousemove
  stage.on('mousemove', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const tool = $tool.get();
    const pos = updateLastCursorPos(stage, $lastCursorPos);
    const selectedLayer = $selectedLayer.get();

    stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.visible(tool === 'brush' || tool === 'eraser');

    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }
    if (!getIsFocused(stage) || !getIsMouseDown(e)) {
      return;
    }

    if (tool === 'brush') {
      if ($isDrawing.get()) {
        // Continue the last line
        maybeAddNextPoint(selectedLayer.id, pos, $lastAddedPoint, $brushSpacingPx, onPointAddedToLine);
      } else {
        // Start a new line
        onBrushLineAdded({
          layerId: selectedLayer.id,
          points: [pos.x, pos.y, pos.x, pos.y],
          color: selectedLayer.type === 'raster_layer' ? $brushColor.get() : DEFAULT_RGBA_COLOR,
        });
        $isDrawing.set(true);
      }
    }

    if (tool === 'eraser') {
      if ($isDrawing.get()) {
        // Continue the last line
        maybeAddNextPoint(selectedLayer.id, pos, $lastAddedPoint, $brushSpacingPx, onPointAddedToLine);
      } else {
        // Start a new line
        onEraserLineAdded({ layerId: selectedLayer.id, points: [pos.x, pos.y, pos.x, pos.y] });
        $isDrawing.set(true);
      }
    }
  });

  //#region mouseleave
  stage.on('mouseleave', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const pos = updateLastCursorPos(stage, $lastCursorPos);
    $isDrawing.set(false);
    $lastCursorPos.set(null);
    $lastMouseDownPos.set(null);
    const selectedLayer = $selectedLayer.get();
    const tool = $tool.get();

    stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.visible(false);

    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }
    if (getIsFocused(stage) && getIsMouseDown(e)) {
      if (tool === 'brush') {
        onPointAddedToLine({ layerId: selectedLayer.id, point: [pos.x, pos.y] });
      }

      if (tool === 'eraser') {
        onPointAddedToLine({ layerId: selectedLayer.id, point: [pos.x, pos.y] });
      }
    }
  });

  stage.on('wheel', (e) => {
    e.evt.preventDefault();
    const tool = $tool.get();
    const selectedLayer = $selectedLayer.get();

    if (tool !== 'brush' && tool !== 'eraser') {
      return;
    }
    if (!selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }
    // Invert the delta if the property is set to true
    let delta = e.evt.deltaY;
    if ($shouldInvertBrushSizeScrollDirection.get()) {
      delta = -delta;
    }

    if (e.evt.ctrlKey || e.evt.metaKey) {
      onBrushSizeChanged(calculateNewBrushSize($brushSize.get(), delta));
    }
  });

  return () => stage.off('mousedown mouseup mousemove mouseenter mouseleave wheel');
};
