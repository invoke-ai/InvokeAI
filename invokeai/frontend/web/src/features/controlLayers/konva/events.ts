import { renderBackgroundLayer } from 'features/controlLayers/konva/renderers/background';
import { renderDocumentBoundsOverlay, scaleToolPreview } from 'features/controlLayers/konva/renderers/previewLayer';
import { getScaledFlooredCursorPosition } from 'features/controlLayers/konva/util';
import type {
  BrushLineAddedArg,
  CanvasEntity,
  CanvasV2State,
  EraserLineAddedArg,
  PointAddedToLineArg,
  RectShapeAddedArg,
  RgbaColor,
  StageAttrs,
  Tool,
} from 'features/controlLayers/store/types';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import { clamp } from 'lodash-es';

import {
  BRUSH_SPACING_TARGET_SCALE,
  CANVAS_SCALE_BY,
  DOCUMENT_FIT_PADDING_PX,
  MAX_BRUSH_SPACING_PX,
  MAX_CANVAS_SCALE,
  MIN_BRUSH_SPACING_PX,
  MIN_CANVAS_SCALE,
} from './constants';
import { PREVIEW_TOOL_GROUP_ID } from './naming';

type Arg = {
  stage: Konva.Stage;
  getToolState: () => CanvasV2State['tool'];
  getCurrentFill: () => RgbaColor;
  setTool: (tool: Tool) => void;
  setToolBuffer: (tool: Tool | null) => void;
  getIsDrawing: () => boolean;
  setIsDrawing: (isDrawing: boolean) => void;
  getIsMouseDown: () => boolean;
  setIsMouseDown: (isMouseDown: boolean) => void;
  getLastMouseDownPos: () => Vector2d | null;
  setLastMouseDownPos: (pos: Vector2d | null) => void;
  getLastCursorPos: () => Vector2d | null;
  setLastCursorPos: (pos: Vector2d | null) => void;
  getLastAddedPoint: () => Vector2d | null;
  setLastAddedPoint: (pos: Vector2d | null) => void;
  setStageAttrs: (attrs: StageAttrs) => void;
  getSelectedEntity: () => CanvasEntity | null;
  getSpaceKey: () => boolean;
  getDocument: () => CanvasV2State['document'];
  onBrushLineAdded: (arg: BrushLineAddedArg, entityType: CanvasEntity['type']) => void;
  onEraserLineAdded: (arg: EraserLineAddedArg, entityType: CanvasEntity['type']) => void;
  onPointAddedToLine: (arg: PointAddedToLineArg, entityType: CanvasEntity['type']) => void;
  onRectShapeAdded: (arg: RectShapeAddedArg, entityType: CanvasEntity['type']) => void;
  onBrushWidthChanged: (size: number) => void;
  onEraserWidthChanged: (size: number) => void;
};

/**
 * Updates the last cursor position atom with the current cursor position, returning the new position or `null` if the
 * cursor is not over the stage.
 * @param stage The konva stage
 * @param setLastCursorPos The callback to store the cursor pos
 */
const updateLastCursorPos = (stage: Konva.Stage, setLastCursorPos: Arg['setLastCursorPos']) => {
  const pos = getScaledFlooredCursorPosition(stage);
  if (!pos) {
    return null;
  }
  setLastCursorPos(pos);
  return pos;
};

const calculateNewBrushSize = (brushSize: number, delta: number) => {
  // This equation was derived by fitting a curve to the desired brush sizes and deltas
  // see https://github.com/invoke-ai/InvokeAI/pull/5542#issuecomment-1915847565
  const targetDelta = Math.sign(delta) * 0.7363 * Math.pow(1.0394, brushSize);
  // This needs to be clamped to prevent the delta from getting too large
  const finalDelta = clamp(targetDelta, -20, 20);
  // The new brush size is also clamped to prevent it from getting too large or small
  const newBrushSize = clamp(brushSize + finalDelta, 1, 500);

  return newBrushSize;
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
  selectedEntity: CanvasEntity,
  currentPos: Vector2d,
  getToolState: Arg['getToolState'],
  getLastAddedPoint: Arg['getLastAddedPoint'],
  setLastAddedPoint: Arg['setLastAddedPoint'],
  onPointAddedToLine: Arg['onPointAddedToLine']
) => {
  if (selectedEntity.type !== 'layer' && selectedEntity.type !== 'regional_guidance') {
    return;
  }
  // Continue the last line
  const lastAddedPoint = getLastAddedPoint();
  const toolState = getToolState();
  const minSpacingPx = clamp(
    toolState.selected === 'brush'
      ? toolState.brush.width * BRUSH_SPACING_TARGET_SCALE
      : toolState.eraser.width * BRUSH_SPACING_TARGET_SCALE,
    MIN_BRUSH_SPACING_PX,
    MAX_BRUSH_SPACING_PX
  );
  if (lastAddedPoint) {
    // Dispatching redux events impacts perf substantially - using brush spacing keeps dispatches to a reasonable number
    if (Math.hypot(lastAddedPoint.x - currentPos.x, lastAddedPoint.y - currentPos.y) < minSpacingPx) {
      return;
    }
  }
  setLastAddedPoint(currentPos);
  onPointAddedToLine(
    {
      id: selectedEntity.id,
      point: [currentPos.x - selectedEntity.x, currentPos.y - selectedEntity.y],
    },
    selectedEntity.type
  );
};

export const setStageEventHandlers = ({
  stage,
  getToolState,
  getCurrentFill,
  setTool,
  setToolBuffer,
  getIsDrawing,
  setIsDrawing,
  getIsMouseDown,
  setIsMouseDown,
  getLastMouseDownPos,
  setLastMouseDownPos,
  getLastCursorPos,
  setLastCursorPos,
  getLastAddedPoint,
  setLastAddedPoint,
  setStageAttrs,
  getSelectedEntity,
  getSpaceKey,
  getDocument,
  onBrushLineAdded,
  onEraserLineAdded,
  onPointAddedToLine,
  onRectShapeAdded,
  onBrushWidthChanged: onBrushSizeChanged,
  onEraserWidthChanged: onEraserSizeChanged,
}: Arg): (() => void) => {
  //#region mouseenter
  stage.on('mouseenter', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const tool = getToolState().selected;
    stage.findOne<Konva.Layer>(`#${PREVIEW_TOOL_GROUP_ID}`)?.visible(tool === 'brush' || tool === 'eraser');
  });

  //#region mousedown
  stage.on('mousedown', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    setIsMouseDown(true);
    const toolState = getToolState();
    const pos = updateLastCursorPos(stage, setLastCursorPos);
    const selectedEntity = getSelectedEntity();
    if (!pos || !selectedEntity) {
      return;
    }
    if (selectedEntity.type !== 'regional_guidance' && selectedEntity.type !== 'layer') {
      return;
    }

    if (getSpaceKey()) {
      // No drawing when space is down - we are panning the stage
      return;
    }

    setIsDrawing(true);
    setLastMouseDownPos(pos);

    if (toolState.selected === 'brush') {
      if (e.evt.shiftKey) {
        const lastAddedPoint = getLastAddedPoint();
        // Create a straight line if holding shift
        if (lastAddedPoint) {
          onBrushLineAdded(
            {
              id: selectedEntity.id,
              points: [
                lastAddedPoint.x - selectedEntity.x,
                lastAddedPoint.y - selectedEntity.y,
                pos.x - selectedEntity.x,
                pos.y - selectedEntity.y,
              ],
              color: getCurrentFill(),
              width: toolState.brush.width,
            },
            selectedEntity.type
          );
        }
      } else {
        onBrushLineAdded(
          {
            id: selectedEntity.id,
            points: [
              pos.x - selectedEntity.x,
              pos.y - selectedEntity.y,
              pos.x - selectedEntity.x,
              pos.y - selectedEntity.y,
            ],
            color: getCurrentFill(),
            width: toolState.brush.width,
          },
          selectedEntity.type
        );
      }
      setLastAddedPoint(pos);
    }

    if (toolState.selected === 'eraser') {
      if (e.evt.shiftKey) {
        // Create a straight line if holding shift
        const lastAddedPoint = getLastAddedPoint();
        if (lastAddedPoint) {
          onEraserLineAdded(
            {
              id: selectedEntity.id,
              points: [
                lastAddedPoint.x - selectedEntity.x,
                lastAddedPoint.y - selectedEntity.y,
                pos.x - selectedEntity.x,
                pos.y - selectedEntity.y,
              ],
              width: toolState.eraser.width,
            },
            selectedEntity.type
          );
        }
      } else {
        onEraserLineAdded(
          {
            id: selectedEntity.id,
            points: [
              pos.x - selectedEntity.x,
              pos.y - selectedEntity.y,
              pos.x - selectedEntity.x,
              pos.y - selectedEntity.y,
            ],
            width: toolState.eraser.width,
          },
          selectedEntity.type
        );
      }
      setLastAddedPoint(pos);
    }
  });

  //#region mouseup
  stage.on('mouseup', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    setIsMouseDown(false);
    const pos = getLastCursorPos();
    const selectedEntity = getSelectedEntity();

    if (!pos || !selectedEntity) {
      return;
    }
    if (selectedEntity.type !== 'regional_guidance' && selectedEntity.type !== 'layer') {
      return;
    }

    if (getSpaceKey()) {
      // No drawing when space is down - we are panning the stage
      return;
    }

    const toolState = getToolState();

    if (toolState.selected === 'rect') {
      const lastMouseDownPos = getLastMouseDownPos();
      if (lastMouseDownPos) {
        onRectShapeAdded(
          {
            id: selectedEntity.id,
            rect: {
              x: Math.min(pos.x, lastMouseDownPos.x),
              y: Math.min(pos.y, lastMouseDownPos.y),
              width: Math.abs(pos.x - lastMouseDownPos.x),
              height: Math.abs(pos.y - lastMouseDownPos.y),
            },
            color: getCurrentFill(),
          },
          selectedEntity.type
        );
      }
    }

    setIsDrawing(false);
    setLastMouseDownPos(null);
  });

  //#region mousemove
  stage.on('mousemove', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const toolState = getToolState();
    const pos = updateLastCursorPos(stage, setLastCursorPos);
    const selectedEntity = getSelectedEntity();

    stage
      .findOne<Konva.Layer>(`#${PREVIEW_TOOL_GROUP_ID}`)
      ?.visible(toolState.selected === 'brush' || toolState.selected === 'eraser');

    if (!pos || !selectedEntity) {
      return;
    }
    if (selectedEntity.type !== 'regional_guidance' && selectedEntity.type !== 'layer') {
      return;
    }

    if (getSpaceKey()) {
      // No drawing when space is down - we are panning the stage
      return;
    }

    if (!getIsMouseDown()) {
      return;
    }

    if (toolState.selected === 'brush') {
      if (getIsDrawing()) {
        // Continue the last line
        maybeAddNextPoint(selectedEntity, pos, getToolState, getLastAddedPoint, setLastAddedPoint, onPointAddedToLine);
      } else {
        // Start a new line
        onBrushLineAdded(
          {
            id: selectedEntity.id,
            points: [
              pos.x - selectedEntity.x,
              pos.y - selectedEntity.y,
              pos.x - selectedEntity.x,
              pos.y - selectedEntity.y,
            ],
            width: toolState.brush.width,
            color: getCurrentFill(),
          },
          selectedEntity.type
        );
        setLastAddedPoint(pos);
        setIsDrawing(true);
      }
    }

    if (toolState.selected === 'eraser') {
      if (getIsDrawing()) {
        // Continue the last line
        maybeAddNextPoint(selectedEntity, pos, getToolState, getLastAddedPoint, setLastAddedPoint, onPointAddedToLine);
      } else {
        // Start a new line
        onEraserLineAdded(
          {
            id: selectedEntity.id,
            points: [
              pos.x - selectedEntity.x,
              pos.y - selectedEntity.y,
              pos.x - selectedEntity.x,
              pos.y - selectedEntity.y,
            ],
            width: toolState.eraser.width,
          },
          selectedEntity.type
        );
        setLastAddedPoint(pos);
        setIsDrawing(true);
      }
    }
  });

  //#region mouseleave
  stage.on('mouseleave', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const pos = updateLastCursorPos(stage, setLastCursorPos);
    setIsDrawing(false);
    setLastCursorPos(null);
    setLastMouseDownPos(null);
    const selectedEntity = getSelectedEntity();
    const toolState = getToolState();

    stage.findOne<Konva.Layer>(`#${PREVIEW_TOOL_GROUP_ID}`)?.visible(false);

    if (!pos || !selectedEntity) {
      return;
    }
    if (selectedEntity.type !== 'regional_guidance' && selectedEntity.type !== 'layer') {
      return;
    }
    if (getSpaceKey()) {
      // No drawing when space is down - we are panning the stage
      return;
    }
    if (getIsMouseDown()) {
      if (toolState.selected === 'brush') {
        onPointAddedToLine({ id: selectedEntity.id, point: [pos.x, pos.y] }, selectedEntity.type);
      }
      if (toolState.selected === 'eraser') {
        onPointAddedToLine({ id: selectedEntity.id, point: [pos.x, pos.y] }, selectedEntity.type);
      }
    }
  });

  //#region wheel
  stage.on('wheel', (e) => {
    e.evt.preventDefault();

    if (e.evt.ctrlKey || e.evt.metaKey) {
      const toolState = getToolState();
      let delta = e.evt.deltaY;
      if (toolState.invertScroll) {
        delta = -delta;
      }
      // Holding ctrl or meta while scrolling changes the brush size
      if (toolState.selected === 'brush') {
        onBrushSizeChanged(calculateNewBrushSize(toolState.brush.width, delta));
      } else if (toolState.selected === 'eraser') {
        onEraserSizeChanged(calculateNewBrushSize(toolState.eraser.width, delta));
      }
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
      setStageAttrs({ ...newPos, width: stage.width(), height: stage.height(), scale: newScale });
      renderBackgroundLayer(stage);
      scaleToolPreview(stage, getToolState());
      renderDocumentBoundsOverlay(stage, getDocument);
    }
  });

  //#region dragmove
  stage.on('dragmove', () => {
    setStageAttrs({
      x: Math.floor(stage.x()),
      y: Math.floor(stage.y()),
      width: stage.width(),
      height: stage.height(),
      scale: stage.scaleX(),
    });
    renderBackgroundLayer(stage);
    renderDocumentBoundsOverlay(stage, getDocument);
  });

  //#region dragend
  stage.on('dragend', () => {
    // Stage position should always be an integer, else we get fractional pixels which are blurry
    setStageAttrs({
      x: Math.floor(stage.x()),
      y: Math.floor(stage.y()),
      width: stage.width(),
      height: stage.height(),
      scale: stage.scaleX(),
    });
  });

  //#region key
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.repeat) {
      return;
    }
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }
    if (e.key === 'Escape') {
      // Cancel shape drawing on escape
      setIsDrawing(false);
      setLastMouseDownPos(null);
    } else if (e.key === ' ') {
      // Select the view tool on space key down
      setToolBuffer(getToolState().selected);
      setTool('view');
    } else if (e.key === 'r') {
      // Fit & center the document on the stage
      const width = stage.width();
      const height = stage.height();
      const document = getDocument();
      const docWidthWithBuffer = document.width + DOCUMENT_FIT_PADDING_PX * 2;
      const docHeightWithBuffer = document.height + DOCUMENT_FIT_PADDING_PX * 2;
      const scale = Math.min(Math.min(width / docWidthWithBuffer, height / docHeightWithBuffer), 1);
      const x = (width - docWidthWithBuffer * scale) / 2 + DOCUMENT_FIT_PADDING_PX * scale;
      const y = (height - docHeightWithBuffer * scale) / 2 + DOCUMENT_FIT_PADDING_PX * scale;
      stage.setAttrs({ x, y, width, height, scaleX: scale, scaleY: scale });
      setStageAttrs({ x, y, width, height, scale });
      scaleToolPreview(stage, getToolState());
      renderBackgroundLayer(stage);
      renderDocumentBoundsOverlay(stage, getDocument);
    }
  };
  window.addEventListener('keydown', onKeyDown);

  const onKeyUp = (e: KeyboardEvent) => {
    if (e.repeat) {
      return;
    }
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }
    if (e.key === ' ') {
      // Revert the tool to the previous tool on space key up
      const toolBuffer = getToolState().selectedBuffer;
      setTool(toolBuffer ?? 'move');
      setToolBuffer(null);
    }
  };
  window.addEventListener('keyup', onKeyUp);

  return () => {
    stage.off('mousedown mouseup mousemove mouseenter mouseleave wheel dragend');
    window.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('keyup', onKeyUp);
  };
};
