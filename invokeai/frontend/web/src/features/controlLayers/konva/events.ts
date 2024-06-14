import { calculateNewBrushSize } from 'features/canvas/hooks/useCanvasZoom';
import { CANVAS_SCALE_BY, MAX_CANVAS_SCALE, MIN_CANVAS_SCALE } from 'features/canvas/util/constants';
import { getScaledFlooredCursorPosition } from 'features/controlLayers/konva/util';
import type {
  AddBrushLineArg,
  AddEraserLineArg,
  AddPointToLineArg,
  AddRectShapeArg,
  LayerData,
  StageAttrs,
  Tool,
} from 'features/controlLayers/store/types';
import { DEFAULT_RGBA_COLOR } from 'features/controlLayers/store/types';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import { clamp } from 'lodash-es';
import type { RgbaColor } from 'react-colorful';

import { PREVIEW_TOOL_GROUP_ID } from './naming';

type Arg = {
  stage: Konva.Stage;
  getTool: () => Tool;
  setTool: (tool: Tool) => void;
  getToolBuffer: () => Tool | null;
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
  getBrushColor: () => RgbaColor;
  getBrushSize: () => number;
  getBrushSpacingPx: () => number;
  getSelectedLayer: () => LayerData | null;
  getShouldInvert: () => boolean;
  getSpaceKey: () => boolean;
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

/**
 * Adds the next point to a line if the cursor has moved far enough from the last point.
 * @param layerId The layer to (maybe) add the point to
 * @param currentPos The current cursor position
 * @param $lastAddedPoint The last added line point as a nanostores atom
 * @param $brushSpacingPx The brush spacing in pixels as a nanostores atom
 * @param onPointAddedToLine The callback to add a point to a line
 */
const maybeAddNextPoint = (
  selectedLayer: LayerData,
  currentPos: Vector2d,
  getLastAddedPoint: Arg['getLastAddedPoint'],
  setLastAddedPoint: Arg['setLastAddedPoint'],
  getBrushSpacingPx: Arg['getBrushSpacingPx'],
  onPointAddedToLine: Arg['onPointAddedToLine']
) => {
  // Continue the last line
  const lastAddedPoint = getLastAddedPoint();
  if (lastAddedPoint) {
    // Dispatching redux events impacts perf substantially - using brush spacing keeps dispatches to a reasonable number
    if (Math.hypot(lastAddedPoint.x - currentPos.x, lastAddedPoint.y - currentPos.y) < getBrushSpacingPx()) {
      return;
    }
  }
  setLastAddedPoint(currentPos);
  onPointAddedToLine({ layerId, point: [currentPos.x - selectedLayer.x, currentPos.y - selectedLayer.y] });
};

export const setStageEventHandlers = ({
  stage,
  getTool,
  setTool,
  getToolBuffer,
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
  getBrushColor,
  getBrushSize,
  getBrushSpacingPx,
  getSelectedLayer,
  getShouldInvert,
  getSpaceKey,
  onBrushLineAdded,
  onEraserLineAdded,
  onPointAddedToLine,
  onRectShapeAdded,
  onBrushSizeChanged,
}: Arg): (() => void) => {
  //#region mouseenter
  stage.on('mouseenter', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    const tool = getTool();
    stage.findOne<Konva.Layer>(`#${PREVIEW_TOOL_GROUP_ID}`)?.visible(tool === 'brush' || tool === 'eraser');
  });

  //#region mousedown
  stage.on('mousedown', (e) => {
    const stage = e.target.getStage();
    if (!stage) {
      return;
    }
    setIsMouseDown(true);
    const tool = getTool();
    const pos = updateLastCursorPos(stage, setLastCursorPos);
    const selectedLayer = getSelectedLayer();
    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }

    if (getSpaceKey()) {
      // No drawing when space is down - we are panning the stage
      return;
    }

    setIsDrawing(true);
    setLastMouseDownPos(pos);

    if (tool === 'brush') {
      onBrushLineAdded({
        layerId: selectedLayer.id,
        points: [pos.x - selectedLayer.x, pos.y - selectedLayer.y, pos.x - selectedLayer.x, pos.y - selectedLayer.y],
        color: selectedLayer.type === 'raster_layer' ? getBrushColor() : DEFAULT_RGBA_COLOR,
      });
    }

    if (tool === 'eraser') {
      onEraserLineAdded({
        layerId: selectedLayer.id,
        points: [pos.x - selectedLayer.x, pos.y - selectedLayer.y, pos.x - selectedLayer.x, pos.y - selectedLayer.y],
      });
    }

    if (tool === 'rect') {
      // Setting the last mouse down pos starts a rect
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
    const selectedLayer = getSelectedLayer();

    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }

    if (getSpaceKey()) {
      // No drawing when space is down - we are panning the stage
      return;
    }

    const tool = getTool();

    if (tool === 'rect') {
      const lastMouseDownPos = getLastMouseDownPos();
      if (lastMouseDownPos) {
        onRectShapeAdded({
          layerId: selectedLayer.id,
          rect: {
            x: Math.min(pos.x, lastMouseDownPos.x),
            y: Math.min(pos.y, lastMouseDownPos.y),
            width: Math.abs(pos.x - lastMouseDownPos.x),
            height: Math.abs(pos.y - lastMouseDownPos.y),
          },
          color: selectedLayer.type === 'raster_layer' ? getBrushColor() : DEFAULT_RGBA_COLOR,
        });
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
    const tool = getTool();
    const pos = updateLastCursorPos(stage, setLastCursorPos);
    const selectedLayer = getSelectedLayer();

    stage.findOne<Konva.Layer>(`#${PREVIEW_TOOL_GROUP_ID}`)?.visible(tool === 'brush' || tool === 'eraser');

    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }

    if (getSpaceKey()) {
      // No drawing when space is down - we are panning the stage
      return;
    }

    if (!getIsMouseDown()) {
      return;
    }

    if (tool === 'brush') {
      if (getIsDrawing()) {
        // Continue the last line
        maybeAddNextPoint(
          selectedLayer.id,
          pos,
          getLastAddedPoint,
          setLastAddedPoint,
          getBrushSpacingPx,
          onPointAddedToLine
        );
      } else {
        // Start a new line
        onBrushLineAdded({
          layerId: selectedLayer.id,
          points: [pos.x - selectedLayer.x, pos.y - selectedLayer.y, pos.x - selectedLayer.x, pos.y - selectedLayer.y],
          color: selectedLayer.type === 'raster_layer' ? getBrushColor() : DEFAULT_RGBA_COLOR,
        });
        setIsDrawing(true);
      }
    }

    if (tool === 'eraser') {
      if (getIsDrawing()) {
        // Continue the last line
        maybeAddNextPoint(
          selectedLayer.id,
          pos,
          getLastAddedPoint,
          setLastAddedPoint,
          getBrushSpacingPx,
          onPointAddedToLine
        );
      } else {
        // Start a new line
        onEraserLineAdded({
          layerId: selectedLayer.id,
          points: [pos.x - selectedLayer.x, pos.y - selectedLayer.y, pos.x - selectedLayer.x, pos.y - selectedLayer.y],
        });
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
    const selectedLayer = getSelectedLayer();
    const tool = getTool();

    stage.findOne<Konva.Layer>(`#${PREVIEW_TOOL_GROUP_ID}`)?.visible(false);

    if (!pos || !selectedLayer) {
      return;
    }
    if (selectedLayer.type !== 'regional_guidance_layer' && selectedLayer.type !== 'raster_layer') {
      return;
    }
    if (getSpaceKey()) {
      // No drawing when space is down - we are panning the stage
      return;
    }
    if (getIsMouseDown()) {
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
      if (getShouldInvert()) {
        delta = -delta;
      }
      // Holding ctrl or meta while scrolling changes the brush size
      onBrushSizeChanged(calculateNewBrushSize(getBrushSize(), delta));
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
    }
  });

  stage.on('dragmove', () => {
    setStageAttrs({
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
    setStageAttrs({
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
      setIsDrawing(false);
      setLastMouseDownPos(null);
    } else if (e.key === ' ') {
      setToolBuffer(getTool());
      setTool('view');
    }
  };
  window.addEventListener('keydown', onKeyDown);

  const onKeyUp = (e: KeyboardEvent) => {
    // Cancel shape drawing on escape
    if (e.repeat) {
      return;
    }
    if (e.key === ' ') {
      const toolBuffer = getToolBuffer();
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
