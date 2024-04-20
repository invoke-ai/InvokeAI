import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  $cursorPosition,
  $isMouseDown,
  $isMouseOver,
  $tool,
  maskLayerLineAdded,
  maskLayerPointsAdded,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useCallback } from 'react';

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

const syncCursorPos = (stage: Konva.Stage) => {
  const pos = getScaledFlooredCursorPosition(stage);
  if (!pos) {
    return null;
  }
  $cursorPosition.set(pos);
  return pos;
};

export const useMouseEvents = () => {
  const dispatch = useAppDispatch();
  const selectedLayerId = useAppSelector((s) => s.regionalPrompts.present.selectedLayerId);
  const tool = useStore($tool);

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
      if (!selectedLayerId) {
        return;
      }
      // const tool = getTool();
      if (tool === 'brush' || tool === 'eraser') {
        dispatch(
          maskLayerLineAdded({
            layerId: selectedLayerId,
            points: [Math.floor(pos.x), Math.floor(pos.y), Math.floor(pos.x), Math.floor(pos.y)],
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
      // const tool = getTool();
      if ((tool === 'brush' || tool === 'eraser') && $isMouseDown.get()) {
        $isMouseDown.set(false);
      }
    },
    [tool]
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
      // const tool = getTool();
      if (getIsFocused(stage) && $isMouseOver.get() && $isMouseDown.get() && (tool === 'brush' || tool === 'eraser')) {
        dispatch(maskLayerPointsAdded({ layerId: selectedLayerId, point: [Math.floor(pos.x), Math.floor(pos.y)] }));
      }
    },
    [dispatch, selectedLayerId, tool]
  );

  const onMouseLeave = useCallback((e: KonvaEventObject<MouseEvent | TouchEvent>) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    $isMouseOver.set(false);
    $isMouseDown.set(false);
    $cursorPosition.set(null);
  }, []);

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
              points: [Math.floor(pos.x), Math.floor(pos.y), Math.floor(pos.x), Math.floor(pos.y)],
              tool,
            })
          );
        }
      }
    },
    [dispatch, selectedLayerId, tool]
  );

  return { onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave };
};
