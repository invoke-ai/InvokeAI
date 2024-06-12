import { calculateNewBrushSize } from 'features/canvas/hooks/useCanvasZoom';
import { CANVAS_SCALE_BY, MAX_CANVAS_SCALE, MIN_CANVAS_SCALE } from 'features/canvas/util/constants';
import { getIsMouseDown, getScaledFlooredCursorPosition, snapPosToStage } from 'features/controlLayers/konva/util';
import type {
  AddBrushLineArg,
  AddEraserLineArg,
  AddPointToLineArg,
  AddRectShapeArg,
  Layer,
  StageAttrs,
  Tool,
} from 'features/controlLayers/store/types';
import { DEFAULT_RGBA_COLOR } from 'features/controlLayers/store/types';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import { clamp } from 'lodash-es';
import type { WritableAtom } from 'nanostores';
import type { RgbaColor } from 'react-colorful';

import { PREVIEW_TOOL_GROUP_ID } from './naming';

type SetStageEventHandlersArg = {
  stage: Konva.Stage;
  $tool: WritableAtom<Tool>;
  $toolBuffer: WritableAtom<Tool | null>;
  $isDrawing: WritableAtom<boolean>;
  $isMouseDown: WritableAtom<boolean>;
  $lastMouseDownPos: WritableAtom<Vector2d | null>;
  $lastCursorPos: WritableAtom<Vector2d | null>;
  $lastAddedPoint: WritableAtom<Vector2d | null>;
  $stageAttrs: WritableAtom<StageAttrs>;
  $brushColor: WritableAtom<RgbaColor>;
  $brushSize: WritableAtom<number>;
  $brushSpacingPx: WritableAtom<number>;
  $selectedLayer: WritableAtom<Layer | null>;
  $shouldInvertBrushSizeScrollDirection: WritableAtom<boolean>;
  $isSpaceDown: WritableAtom<boolean>;
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
      return;
    }
  }
  $lastAddedPoint.set(currentPos);
  onPointAddedToLine({ layerId, point: [currentPos.x, currentPos.y] });
};

export const setStageEventHandlers = ({
  stage,
  $tool,
  $toolBuffer,
  $isDrawing,
  $isMouseDown,
  $lastMouseDownPos,
  $lastCursorPos,
  $lastAddedPoint,
  $stageAttrs,
  $brushColor,
  $brushSize,
  $brushSpacingPx,
  $selectedLayer,
  $shouldInvertBrushSizeScrollDirection,
  $isSpaceDown,
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
    stage.findOne<Konva.Layer>(`#${PREVIEW_TOOL_GROUP_ID}`)?.visible(tool === 'brush' || tool === 'eraser');
  });

  //#region mousedown
  stage.on('mousedown', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    $isMouseDown.set(true);
    const tool = $tool.get();
    const pos = updateLastCursorPos(stage, $lastCursorPos);
    const selectedLayer = $selectedLayer.get();
    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }

    if ($isSpaceDown.get()) {
      // No drawing when space is down - we are panning the stage
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
    $isMouseDown.set(false);
    const pos = $lastCursorPos.get();
    const selectedLayer = $selectedLayer.get();

    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }

    if ($isSpaceDown.get()) {
      // No drawing when space is down - we are panning the stage
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

    stage.findOne<Konva.Layer>(`#${PREVIEW_TOOL_GROUP_ID}`)?.visible(tool === 'brush' || tool === 'eraser');

    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }

    if ($isSpaceDown.get()) {
      // No drawing when space is down - we are panning the stage
      return;
    }

    if (!getIsMouseDown(e)) {
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

    stage.findOne<Konva.Layer>(`#${PREVIEW_TOOL_GROUP_ID}`)?.visible(false);

    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }
    if ($isSpaceDown.get()) {
      // No drawing when space is down - we are panning the stage
      return;
    }
    if (getIsMouseDown(e)) {
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

    if (e.evt.ctrlKey || e.evt.metaKey) {
      let delta = e.evt.deltaY;
      if ($shouldInvertBrushSizeScrollDirection.get()) {
        delta = -delta;
      }
      // Holding ctrl or meta while scrolling changes the brush size
      onBrushSizeChanged(calculateNewBrushSize($brushSize.get(), delta));
    } else {
      // We need the absolute cursor position - not the scaled position
      const cursorPos = stage.getPointerPosition();
      if (!cursorPos) {
        return;
      }
      // Stage's x and y scale are always the same
      const stageScale = stage.scaleX();
      // When wheeling on trackpad, e.evt.ctrlKey is true - in that case, let's reverse the direction
      const delta = e.evt.ctrlKey ? -e.evt.deltaY : e.evt.deltaY;
      const mousePointTo = {
        x: (cursorPos.x - stage.x()) / stageScale,
        y: (cursorPos.y - stage.y()) / stageScale,
      };
      const newScale = clamp(stageScale * CANVAS_SCALE_BY ** delta, MIN_CANVAS_SCALE, MAX_CANVAS_SCALE);
      const newPos = {
        x: cursorPos.x - mousePointTo.x * newScale,
        y: cursorPos.y - mousePointTo.y * newScale,
      };

      stage.scaleX(newScale);
      stage.scaleY(newScale);
      stage.position(newPos);
      $stageAttrs.set({ ...newPos, width: stage.width(), height: stage.height(), scale: newScale });
    }
  });

  stage.on('dragmove', () => {
    $stageAttrs.set({
      x: stage.x(),
      y: stage.y(),
      width: stage.width(),
      height: stage.height(),
      scale: stage.scaleX(),
    });
  });

  stage.on('dragend', () => {
    // Stage position should always be an integer, else we get fractional pixels which are blurry
    stage.x(Math.floor(stage.x()));
    stage.y(Math.floor(stage.y()));
    $stageAttrs.set({
      x: stage.x(),
      y: stage.y(),
      width: stage.width(),
      height: stage.height(),
      scale: stage.scaleX(),
    });
  });

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.repeat) {
      return;
    }
    // Cancel shape drawing on escape
    if (e.key === 'Escape') {
      $isDrawing.set(false);
      $lastMouseDownPos.set(null);
    } else if (e.key === ' ') {
      $toolBuffer.set($tool.get());
      $tool.set('view');
    }
  };
  window.addEventListener('keydown', onKeyDown);

  const onKeyUp = (e: KeyboardEvent) => {
    // Cancel shape drawing on escape
    if (e.repeat) {
      return;
    }
    if (e.key === ' ') {
      const toolBuffer = $toolBuffer.get();
      $tool.set(toolBuffer ?? 'move');
      $toolBuffer.set(null);
    }
  };
  window.addEventListener('keyup', onKeyUp);

  return () => {
    stage.off('mousedown mouseup mousemove mouseenter mouseleave wheel dragend');
    window.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('keyup', onKeyUp);
  };
};
