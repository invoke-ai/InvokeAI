import { calculateNewBrushSize } from 'features/canvas/hooks/useCanvasZoom';
import {
  getIsFocused,
  getIsMouseDown,
  getScaledFlooredCursorPosition,
  snapPosToStage,
} from 'features/controlLayers/konva/util';
import type { AddLineArg, AddPointToLineArg, AddRectArg, Layer, Tool } from 'features/controlLayers/store/types';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import type { WritableAtom } from 'nanostores';

import { TOOL_PREVIEW_LAYER_ID } from './naming';

type SetStageEventHandlersArg = {
  stage: Konva.Stage;
  $tool: WritableAtom<Tool>;
  $isDrawing: WritableAtom<boolean>;
  $lastMouseDownPos: WritableAtom<Vector2d | null>;
  $lastCursorPos: WritableAtom<Vector2d | null>;
  $lastAddedPoint: WritableAtom<Vector2d | null>;
  $brushSize: WritableAtom<number>;
  $brushSpacingPx: WritableAtom<number>;
  $selectedLayerId: WritableAtom<string | null>;
  $selectedLayerType: WritableAtom<Layer['type'] | null>;
  $shouldInvertBrushSizeScrollDirection: WritableAtom<boolean>;
  onRGLayerLineAdded: (arg: AddLineArg) => void;
  onRGLayerPointAddedToLine: (arg: AddPointToLineArg) => void;
  onRGLayerRectAdded: (arg: AddRectArg) => void;
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
  $brushSize,
  $brushSpacingPx,
  $selectedLayerId,
  $selectedLayerType,
  $shouldInvertBrushSizeScrollDirection,
  onRGLayerLineAdded,
  onRGLayerPointAddedToLine,
  onRGLayerRectAdded,
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
    const selectedLayerId = $selectedLayerId.get();
    const selectedLayerType = $selectedLayerType.get();
    if (!pos || !selectedLayerId || selectedLayerType !== 'regional_guidance_layer') {
      return;
    }
    if (tool === 'brush' || tool === 'eraser') {
      onRGLayerLineAdded({
        layerId: selectedLayerId,
        points: [pos.x, pos.y, pos.x, pos.y],
        tool,
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
    const selectedLayerId = $selectedLayerId.get();
    const selectedLayerType = $selectedLayerType.get();

    if (!pos || !selectedLayerId || selectedLayerType !== 'regional_guidance_layer') {
      return;
    }
    const lastPos = $lastMouseDownPos.get();
    const tool = $tool.get();
    if (lastPos && selectedLayerId && tool === 'rect') {
      const snappedPos = snapPosToStage(pos, stage);
      onRGLayerRectAdded({
        layerId: selectedLayerId,
        rect: {
          x: Math.min(snappedPos.x, lastPos.x),
          y: Math.min(snappedPos.y, lastPos.y),
          width: Math.abs(snappedPos.x - lastPos.x),
          height: Math.abs(snappedPos.y - lastPos.y),
        },
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
    const selectedLayerId = $selectedLayerId.get();
    const selectedLayerType = $selectedLayerType.get();

    stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.visible(tool === 'brush' || tool === 'eraser');

    if (!pos || !selectedLayerId || selectedLayerType !== 'regional_guidance_layer') {
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
        onRGLayerPointAddedToLine({ layerId: selectedLayerId, point: [pos.x, pos.y] });
      } else {
        // Start a new line
        onRGLayerLineAdded({ layerId: selectedLayerId, points: [pos.x, pos.y, pos.x, pos.y], tool });
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
    const selectedLayerId = $selectedLayerId.get();
    const selectedLayerType = $selectedLayerType.get();
    const tool = $tool.get();

    stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.visible(false);

    if (!pos || !selectedLayerId || selectedLayerType !== 'regional_guidance_layer') {
      return;
    }
    if (getIsFocused(stage) && getIsMouseDown(e) && (tool === 'brush' || tool === 'eraser')) {
      onRGLayerPointAddedToLine({ layerId: selectedLayerId, point: [pos.x, pos.y] });
    }
  });

  stage.on('wheel', (e) => {
    e.evt.preventDefault();
    const selectedLayerType = $selectedLayerType.get();
    const tool = $tool.get();
    if (selectedLayerType !== 'regional_guidance_layer' || (tool !== 'brush' && tool !== 'eraser')) {
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
