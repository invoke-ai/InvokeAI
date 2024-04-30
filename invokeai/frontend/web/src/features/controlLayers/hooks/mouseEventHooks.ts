import { $ctrl, $meta } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { calculateNewBrushSize } from 'features/canvas/hooks/useCanvasZoom';
import {
  $cursorPosition,
  $isMouseDown,
  $isMouseOver,
  $lastMouseDownPos,
  $tool,
  brushSizeChanged,
  maskLayerLineAdded,
  maskLayerPointsAdded,
  maskLayerRectAdded,
} from 'features/controlLayers/store/regionalPromptsSlice';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Vector2d } from 'konva/lib/types';
import { useCallback, useRef } from 'react';

const getIsFocused = (stage: Konva.Stage) => {
  return stage.container().contains(document.activeElement);
};

export const getScaledFlooredCursorPosition = (stage: Konva.Stage) => {
  const pointerPosition = stage.getPointerPosition();
  const stageTransform = stage.getAbsoluteTransform().copy();
  if (!pointerPosition || !stageTransform) {
    return;
  }
  const scaledCursorPosition = stageTransform.invert().point(pointerPosition);
  return {
    x: Math.floor(scaledCursorPosition.x),
    y: Math.floor(scaledCursorPosition.y),
  };
};

const syncCursorPos = (stage: Konva.Stage): Vector2d | null => {
  const pos = getScaledFlooredCursorPosition(stage);
  if (!pos) {
    return null;
  }
  $cursorPosition.set(pos);
  return pos;
};

const BRUSH_SPACING = 20;

export const useMouseEvents = () => {
  const dispatch = useAppDispatch();
  const selectedLayerId = useAppSelector((s) => s.regionalPrompts.present.selectedLayerId);
  const tool = useStore($tool);
  const lastCursorPosRef = useRef<[number, number] | null>(null);
  const shouldInvertBrushSizeScrollDirection = useAppSelector((s) => s.canvas.shouldInvertBrushSizeScrollDirection);
  const brushSize = useAppSelector((s) => s.regionalPrompts.present.brushSize);

  const onMouseDown = useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      const stage = e.target.getStage();
      if (!stage) {
        return;
      }
      const pos = syncCursorPos(stage);
      if (!pos) {
        return;
      }
      $isMouseDown.set(true);
      $lastMouseDownPos.set(pos);
      if (!selectedLayerId) {
        return;
      }
      if (tool === 'brush' || tool === 'eraser') {
        dispatch(
          maskLayerLineAdded({
            layerId: selectedLayerId,
            points: [pos.x, pos.y, pos.x, pos.y],
            tool,
          })
        );
      }
    },
    [dispatch, selectedLayerId, tool]
  );

  const onMouseUp = useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      const stage = e.target.getStage();
      if (!stage) {
        return;
      }
      $isMouseDown.set(false);
      const pos = $cursorPosition.get();
      const lastPos = $lastMouseDownPos.get();
      const tool = $tool.get();
      if (pos && lastPos && selectedLayerId && tool === 'rect') {
        dispatch(
          maskLayerRectAdded({
            layerId: selectedLayerId,
            rect: {
              x: Math.min(pos.x, lastPos.x),
              y: Math.min(pos.y, lastPos.y),
              width: Math.abs(pos.x - lastPos.x),
              height: Math.abs(pos.y - lastPos.y),
            },
          })
        );
      }
      $lastMouseDownPos.set(null);
    },
    [dispatch, selectedLayerId]
  );

  const onMouseMove = useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      const stage = e.target.getStage();
      if (!stage) {
        return;
      }
      const pos = syncCursorPos(stage);
      if (!pos || !selectedLayerId) {
        return;
      }
      if (getIsFocused(stage) && $isMouseOver.get() && $isMouseDown.get() && (tool === 'brush' || tool === 'eraser')) {
        if (lastCursorPosRef.current) {
          // Dispatching redux events impacts perf substantially - using brush spacing keeps dispatches to a reasonable number
          if (Math.hypot(lastCursorPosRef.current[0] - pos.x, lastCursorPosRef.current[1] - pos.y) < BRUSH_SPACING) {
            return;
          }
        }
        lastCursorPosRef.current = [pos.x, pos.y];
        dispatch(maskLayerPointsAdded({ layerId: selectedLayerId, point: lastCursorPosRef.current }));
      }
    },
    [dispatch, selectedLayerId, tool]
  );

  const onMouseLeave = useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      const stage = e.target.getStage();
      if (!stage) {
        return;
      }
      const pos = syncCursorPos(stage);
      if (
        pos &&
        selectedLayerId &&
        getIsFocused(stage) &&
        $isMouseOver.get() &&
        $isMouseDown.get() &&
        (tool === 'brush' || tool === 'eraser')
      ) {
        dispatch(maskLayerPointsAdded({ layerId: selectedLayerId, point: [pos.x, pos.y] }));
      }
      $isMouseOver.set(false);
      $isMouseDown.set(false);
      $cursorPosition.set(null);
    },
    [selectedLayerId, tool, dispatch]
  );

  const onMouseEnter = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      const stage = e.target.getStage();
      if (!stage) {
        return;
      }
      $isMouseOver.set(true);
      const pos = syncCursorPos(stage);
      if (!pos) {
        return;
      }
      if (!getIsFocused(stage)) {
        return;
      }
      if (e.evt.buttons !== 1) {
        $isMouseDown.set(false);
      } else {
        $isMouseDown.set(true);
        if (!selectedLayerId) {
          return;
        }
        if (tool === 'brush' || tool === 'eraser') {
          dispatch(
            maskLayerLineAdded({
              layerId: selectedLayerId,
              points: [pos.x, pos.y, pos.x, pos.y],
              tool,
            })
          );
        }
      }
    },
    [dispatch, selectedLayerId, tool]
  );

  const onMouseWheel = useCallback(
    (e: KonvaEventObject<WheelEvent>) => {
      e.evt.preventDefault();

      // checking for ctrl key is pressed or not,
      // so that brush size can be controlled using ctrl + scroll up/down

      // Invert the delta if the property is set to true
      let delta = e.evt.deltaY;
      if (shouldInvertBrushSizeScrollDirection) {
        delta = -delta;
      }

      if ($ctrl.get() || $meta.get()) {
        dispatch(brushSizeChanged(calculateNewBrushSize(brushSize, delta)));
      }
    },
    [shouldInvertBrushSizeScrollDirection, brushSize, dispatch]
  );

  return { onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave, onMouseWheel };
};
