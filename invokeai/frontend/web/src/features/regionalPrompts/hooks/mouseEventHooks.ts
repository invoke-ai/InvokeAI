import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import {
  $cursorPosition,
  $isMouseDown,
  $isMouseOver,
  $tool,
  rpLayerLineAdded,
  rpLayerPointsAdded,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useCallback } from 'react';

const getIsFocused = (stage: Konva.Stage) => {
  return stage.container().contains(document.activeElement);
};

const syncCursorPos = (stage: Konva.Stage) => {
  const pos = getScaledCursorPosition(stage);
  if (!pos) {
    return null;
  }
  $cursorPosition.set(pos);
  return pos;
};

export const useMouseEvents = () => {
  const dispatch = useAppDispatch();
  const selectedLayer = useAppSelector((s) => s.regionalPrompts.present.selectedLayer);
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
      if (!selectedLayer) {
        return;
      }
      // const tool = getTool();
      if (tool === 'brush' || tool === 'eraser') {
        dispatch(rpLayerLineAdded({ layerId: selectedLayer, points: [pos.x, pos.y, pos.x, pos.y], tool }));
      }
    },
    [dispatch, selectedLayer, tool]
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
      if (!pos || !selectedLayer) {
        return;
      }
      // const tool = getTool();
      if (getIsFocused(stage) && $isMouseOver.get() && $isMouseDown.get() && (tool === 'brush' || tool === 'eraser')) {
        dispatch(rpLayerPointsAdded({ layerId: selectedLayer, point: [pos.x, pos.y] }));
      }
    },
    [dispatch, selectedLayer, tool]
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
        if (!selectedLayer) {
          return;
        }
        if (tool === 'brush' || tool === 'eraser') {
          dispatch(rpLayerLineAdded({ layerId: selectedLayer, points: [pos.x, pos.y, pos.x, pos.y], tool }));
        }
      }
    },
    [dispatch, selectedLayer, tool]
  );

  return { onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave };
};
