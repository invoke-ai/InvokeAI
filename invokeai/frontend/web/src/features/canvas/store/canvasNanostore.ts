import type { Vector2d } from 'konva/lib/types';
import { atom, computed } from 'nanostores';

export const $cursorPosition = atom<Vector2d | null>(null);
export const $isDrawing = atom<boolean>(false);
export const $isMouseOverBoundingBox = atom<boolean>(false);
export const $isMoveBoundingBoxKeyHeld = atom<boolean>(false);
export const $isMoveStageKeyHeld = atom<boolean>(false);
export const $isMovingBoundingBox = atom<boolean>(false);
export const $isMovingStage = atom<boolean>(false);
export const $isTransformingBoundingBox = atom<boolean>(false);
export const $isModifyingBoundingBox = computed(
  [$isTransformingBoundingBox, $isMovingBoundingBox],
  (isTransformingBoundingBox, isMovingBoundingBox) =>
    isTransformingBoundingBox || isMovingBoundingBox
);

export const resetCanvasInteractionState = () => {
  $cursorPosition.set(null);
  $isDrawing.set(false);
  $isMouseOverBoundingBox.set(false);
  $isMoveBoundingBoxKeyHeld.set(false);
  $isMoveStageKeyHeld.set(false);
  $isMovingBoundingBox.set(false);
  $isMovingStage.set(false);
};

export const setCursorPosition = (cursorPosition: Vector2d | null) => {
  $cursorPosition.set(cursorPosition);
};

export const setIsDrawing = (isDrawing: boolean) => {
  $isDrawing.set(isDrawing);
};

export const setIsMouseOverBoundingBox = (isMouseOverBoundingBox: boolean) => {
  $isMouseOverBoundingBox.set(isMouseOverBoundingBox);
};

export const setIsMoveBoundingBoxKeyHeld = (
  isMoveBoundingBoxKeyHeld: boolean
) => {
  $isMoveBoundingBoxKeyHeld.set(isMoveBoundingBoxKeyHeld);
};

export const setIsMoveStageKeyHeld = (isMoveStageKeyHeld: boolean) => {
  $isMoveStageKeyHeld.set(isMoveStageKeyHeld);
};

export const setIsMovingBoundingBox = (isMovingBoundingBox: boolean) => {
  $isMovingBoundingBox.set(isMovingBoundingBox);
};

export const setIsMovingStage = (isMovingStage: boolean) => {
  $isMovingStage.set(isMovingStage);
};

export const setIsTransformingBoundingBox = (
  isTransformingBoundingBox: boolean
) => {
  $isTransformingBoundingBox.set(isTransformingBoundingBox);
};

export const resetToolInteractionState = () => {
  setIsTransformingBoundingBox(false);
  setIsMouseOverBoundingBox(false);
  setIsMovingBoundingBox(false);
  setIsMovingStage(false);
};

export const setCanvasInteractionStateMouseOut = () => {
  setCursorPosition(null);
  setIsDrawing(false);
  setIsMouseOverBoundingBox(false);
  setIsMovingBoundingBox(false);
  setIsTransformingBoundingBox(false);
};
