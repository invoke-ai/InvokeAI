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

const syncCursorPos = (stage: Konva.Stage, $lastCursorPos: WritableAtom<Vector2d | null>) => {
  const pos = getScaledFlooredCursorPosition(stage);
  if (!pos) {
    return null;
  }
  $lastCursorPos.set(pos);
  return pos;
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
  stage.on('mouseenter', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const tool = $tool.get();
    stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.visible(tool === 'brush' || tool === 'eraser');
  });

  stage.on('mousedown', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const tool = $tool.get();
    const pos = syncCursorPos(stage, $lastCursorPos);
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
    } else if (tool === 'eraser') {
      onEraserLineAdded({
        layerId: selectedLayer.id,
        points: [pos.x, pos.y, pos.x, pos.y],
      });
      $isDrawing.set(true);
      $lastMouseDownPos.set(pos);
    } else if (tool === 'rect') {
      $lastMouseDownPos.set(snapPosToStage(pos, stage));
    }
  });

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
    const lastPos = $lastMouseDownPos.get();
    const tool = $tool.get();
    if (lastPos && selectedLayer.id && tool === 'rect') {
      const snappedPos = snapPosToStage(pos, stage);
      onRectShapeAdded({
        layerId: selectedLayer.id,
        rect: {
          x: Math.min(snappedPos.x, lastPos.x),
          y: Math.min(snappedPos.y, lastPos.y),
          width: Math.abs(snappedPos.x - lastPos.x),
          height: Math.abs(snappedPos.y - lastPos.y),
        },
        color: selectedLayer.type === 'raster_layer' ? $brushColor.get() : DEFAULT_RGBA_COLOR,
      });
    }
    $isDrawing.set(false);
    $lastMouseDownPos.set(null);
  });

  stage.on('mousemove', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const tool = $tool.get();
    const pos = syncCursorPos(stage, $lastCursorPos);
    const selectedLayer = $selectedLayer.get();

    stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.visible(tool === 'brush' || tool === 'eraser');

    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }
    if (getIsFocused(stage) && getIsMouseDown(e) && (tool === 'brush' || tool === 'eraser')) {
      if ($isDrawing.get()) {
        // Continue the last line
        const lastAddedPoint = $lastAddedPoint.get();
        if (lastAddedPoint) {
          // Dispatching redux events impacts perf substantially - using brush spacing keeps dispatches to a reasonable number
          if (Math.hypot(lastAddedPoint.x - pos.x, lastAddedPoint.y - pos.y) < $brushSpacingPx.get()) {
            return;
          }
        }
        $lastAddedPoint.set({ x: pos.x, y: pos.y });
        onPointAddedToLine({ layerId: selectedLayer.id, point: [pos.x, pos.y] });
      } else {
        if (tool === 'brush') {
          // Start a new line
          onBrushLineAdded({
            layerId: selectedLayer.id,
            points: [pos.x, pos.y, pos.x, pos.y],
            color: selectedLayer.type === 'raster_layer' ? $brushColor.get() : DEFAULT_RGBA_COLOR,
          });
        } else if (tool === 'eraser') {
          onEraserLineAdded({
            layerId: selectedLayer.id,
            points: [pos.x, pos.y, pos.x, pos.y],
          });
        }
      }
      $isDrawing.set(true);
    }
  });

  stage.on('mouseleave', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const pos = syncCursorPos(stage, $lastCursorPos);
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
    if (getIsFocused(stage) && getIsMouseDown(e) && (tool === 'brush' || tool === 'eraser')) {
      onPointAddedToLine({ layerId: selectedLayer.id, point: [pos.x, pos.y] });
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
