import { getStore } from 'app/store/nanostores/store';
import { useAppDispatch } from 'app/store/storeHooks';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import {
  $cursorPosition,
  $isMouseDown,
  $isMouseOver,
  lineAdded,
  pointsAdded,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useCallback } from 'react';

const getTool = () => getStore().getState().regionalPrompts.tool;

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

export const useMouseDown = () => {
  const dispatch = useAppDispatch();
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
      const tool = getTool();
      if (tool === 'brush' || tool === 'eraser') {
        dispatch(lineAdded([pos.x, pos.y]));
      }
    },
    [dispatch]
  );
  return onMouseDown;
};

export const useMouseUp = () => {
  const dispatch = useAppDispatch();
  const onMouseUp = useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      const stage = e.target.getStage();
      if (!stage) {
        return;
      }
      const tool = getTool();
      if ((tool === 'brush' || tool === 'eraser') && $isMouseDown.get()) {
        // Add another point to the last line.
        $isMouseDown.set(false);
        const pos = syncCursorPos(stage);
        if (!pos) {
          return;
        }
        dispatch(pointsAdded([pos.x, pos.y]));
      }
    },
    [dispatch]
  );
  return onMouseUp;
};

export const useMouseMove = () => {
  const dispatch = useAppDispatch();
  const onMouseMove = useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      const stage = e.target.getStage();
      if (!stage) {
        return;
      }
      const pos = syncCursorPos(stage);
      if (!pos) {
        return;
      }
      const tool = getTool();
      if (getIsFocused(stage) && $isMouseOver.get() && $isMouseDown.get() && (tool === 'brush' || tool === 'eraser')) {
        dispatch(pointsAdded([pos.x, pos.y]));
      }
    },
    [dispatch]
  );
  return onMouseMove;
};

export const useMouseLeave = () => {
  const onMouseLeave = useCallback((e: KonvaEventObject<MouseEvent | TouchEvent>) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    $isMouseOver.set(false);
    $isMouseDown.set(false);
    $cursorPosition.set(null);
  }, []);
  return onMouseLeave;
};

export const useMouseEnter = () => {
  const dispatch = useAppDispatch();
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
        const tool = getTool();
        if (tool === 'brush' || tool === 'eraser') {
          dispatch(lineAdded([pos.x, pos.y]));
        }
      }
    },
    [dispatch]
  );
  return onMouseEnter;
};
