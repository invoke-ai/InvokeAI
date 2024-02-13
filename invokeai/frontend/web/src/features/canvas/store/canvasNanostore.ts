import type { CanvasTool } from 'features/canvas/store/canvasTypes';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import { atom, computed } from 'nanostores';

export const $cursorPosition = atom<Vector2d | null>(null);
export const $tool = atom<CanvasTool>('move');
export const $toolStash = atom<CanvasTool | null>(null);
export const $isDrawing = atom<boolean>(false);
export const $isMouseOverBoundingBox = atom<boolean>(false);
export const $isMoveBoundingBoxKeyHeld = atom<boolean>(false);
export const $isMoveStageKeyHeld = atom<boolean>(false);
export const $isMovingBoundingBox = atom<boolean>(false);
export const $isMovingStage = atom<boolean>(false);
export const $isTransformingBoundingBox = atom<boolean>(false);
export const $isMouseOverBoundingBoxOutline = atom<boolean>(false);
export const $isModifyingBoundingBox = computed(
  [$isTransformingBoundingBox, $isMovingBoundingBox],
  (isTransformingBoundingBox, isMovingBoundingBox) => isTransformingBoundingBox || isMovingBoundingBox
);

export const resetCanvasInteractionState = () => {
  $cursorPosition.set(null);
  $isDrawing.set(false);
  $isMoveBoundingBoxKeyHeld.set(false);
  $isMoveStageKeyHeld.set(false);
  $isMovingBoundingBox.set(false);
  $isMovingStage.set(false);
};

export const resetToolInteractionState = () => {
  $isTransformingBoundingBox.set(false);
  $isMovingBoundingBox.set(false);
  $isMovingStage.set(false);
};

export const setCanvasInteractionStateMouseOut = () => {
  $cursorPosition.set(null);
};
export const $canvasBaseLayer = atom<Konva.Layer | null>(null);
export const $canvasStage = atom<Konva.Stage | null>(null);
