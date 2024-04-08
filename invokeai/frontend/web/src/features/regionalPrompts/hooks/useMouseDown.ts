import { useAppDispatch } from 'app/store/storeHooks';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import {
  $cursorPosition,
  $isMouseDown,
  $isMouseOver,
  $tool,
  lineAdded,
  pointsAdded,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { MutableRefObject } from 'react';
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

export const useMouseDown = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const onMouseDown = useCallback(
    (_e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!stageRef.current) {
        return;
      }
      console.log('Mouse down');
      const pos = syncCursorPos(stageRef.current);
      if (!pos) {
        return;
      }
      $isMouseDown.set(true);
      if ($tool.get() === 'brush') {
        dispatch(lineAdded([pos.x, pos.y]));
      }
    },
    [dispatch, stageRef]
  );
  return onMouseDown;
};

export const useMouseUp = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const onMouseUp = useCallback(
    (_e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!stageRef.current) {
        return;
      }
      console.log('Mouse up');
      if ($tool.get() === 'brush' && $isMouseDown.get()) {
        // Add another point to the last line.
        $isMouseDown.set(false);
        const pos = syncCursorPos(stageRef.current);
        if (!pos) {
          return;
        }
        dispatch(pointsAdded([pos.x, pos.y]));
      }
    },
    [dispatch, stageRef]
  );
  return onMouseUp;
};

export const useMouseMove = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const onMouseMove = useCallback(
    (_e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!stageRef.current) {
        return;
      }
      console.log('Mouse move');
      const pos = syncCursorPos(stageRef.current);
      if (!pos) {
        return;
      }
      if (getIsFocused(stageRef.current) && $isMouseOver.get() && $isMouseDown.get() && $tool.get() === 'brush') {
        dispatch(pointsAdded([pos.x, pos.y]));
      }
    },
    [dispatch, stageRef]
  );
  return onMouseMove;
};

export const useMouseLeave = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const onMouseLeave = useCallback(
    (_e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!stageRef.current) {
        return;
      }
      console.log('Mouse leave');
      $isMouseOver.set(false);
      $isMouseDown.set(false);
      $cursorPosition.set(null);
    },
    [stageRef]
  );
  return onMouseLeave;
};

export const useMouseEnter = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const onMouseEnter = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!stageRef.current) {
        return;
      }
      console.log('Mouse enter');
      $isMouseOver.set(true);
      const pos = syncCursorPos(stageRef.current);
      if (!pos) {
        return;
      }
      if (!getIsFocused(stageRef.current)) {
        return;
      }
      if (e.evt.buttons !== 1) {
        $isMouseDown.set(false);
      } else {
        $isMouseDown.set(true);
        if ($tool.get() === 'brush') {
          dispatch(lineAdded([pos.x, pos.y]));
        }
      }
    },
    [dispatch, stageRef]
  );
  return onMouseEnter;
};
