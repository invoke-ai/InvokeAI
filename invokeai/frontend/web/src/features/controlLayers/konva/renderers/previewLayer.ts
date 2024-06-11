import { roundToMultiple, roundToMultipleMin } from 'common/util/roundDownToMultiple';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import {
  BBOX_SELECTED_STROKE,
  BRUSH_BORDER_INNER_COLOR,
  BRUSH_BORDER_OUTER_COLOR,
} from 'features/controlLayers/konva/constants';
import {
  PREVIEW_BRUSH_BORDER_INNER_ID,
  PREVIEW_BRUSH_BORDER_OUTER_ID,
  PREVIEW_BRUSH_FILL_ID,
  PREVIEW_BRUSH_GROUP_ID,
  PREVIEW_GENERATION_BBOX_DUMMY_RECT,
  PREVIEW_GENERATION_BBOX_GROUP,
  PREVIEW_GENERATION_BBOX_TRANSFORMER,
  PREVIEW_LAYER_ID,
  PREVIEW_RECT_ID,
  PREVIEW_TOOL_GROUP_ID,
} from 'features/controlLayers/konva/naming';
import { selectRenderableLayers, snapPosToStage } from 'features/controlLayers/konva/util';
import type { Layer, RgbaColor, Tool } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { IRect, Vector2d } from 'konva/lib/types';
import { assert } from 'tsafe';

/**
 * Creates the singleton preview layer and all its objects.
 * @param stage The konva stage
 */
const getPreviewLayer = (stage: Konva.Stage): Konva.Layer => {
  let previewLayer = stage.findOne<Konva.Layer>(`#${PREVIEW_LAYER_ID}`);
  if (previewLayer) {
    return previewLayer;
  }
  // Initialize the preview layer & add to the stage
  previewLayer = new Konva.Layer({ id: PREVIEW_LAYER_ID, listening: true });
  stage.add(previewLayer);
  return previewLayer;
};

export const getBboxPreviewGroup = (
  stage: Konva.Stage,
  getBbox: () => IRect,
  onBboxTransformed: (bbox: IRect) => void,
  getShiftKey: () => boolean,
  getMetaKey: () => boolean,
  getAltKey: () => boolean
): Konva.Group => {
  const previewLayer = getPreviewLayer(stage);
  let bboxPreviewGroup = previewLayer.findOne<Konva.Group>(`#${PREVIEW_GENERATION_BBOX_GROUP}`);

  if (bboxPreviewGroup) {
    return bboxPreviewGroup;
  }
  console.log('creating new bbox');

  // Use a transformer for the generation bbox. Transformers need some shape to transform, we will use a fully
  // transparent rect for this purpose.
  bboxPreviewGroup = new Konva.Group({ id: PREVIEW_GENERATION_BBOX_GROUP });
  const bboxRect = new Konva.Rect({
    id: PREVIEW_GENERATION_BBOX_DUMMY_RECT,
    listening: true,
    strokeEnabled: false,
    draggable: true,
    fill: 'rgba(255,0,0,0.3)',
    ...getBbox(),
  });
  bboxRect.on('dragmove', () => {
    const gridSize = getMetaKey() ? 8 : 64;
    const oldBbox = getBbox();
    const newBbox: IRect = {
      ...oldBbox,
      x: roundToMultiple(bboxRect.x(), gridSize),
      y: roundToMultiple(bboxRect.y(), gridSize),
    };
    bboxRect.setAttrs(newBbox);
    if (oldBbox.x !== newBbox.x || oldBbox.y !== newBbox.y) {
      onBboxTransformed(newBbox);
    }
  });
  const bboxTransformer = new Konva.Transformer({
    id: PREVIEW_GENERATION_BBOX_TRANSFORMER,
    borderDash: [5, 5],
    borderStroke: 'rgba(212,216,234,1)',
    borderEnabled: true,
    rotateEnabled: false,
    keepRatio: false,
    ignoreStroke: true,
    listening: false,
    flipEnabled: false,
    anchorFill: 'rgba(212,216,234,1)',
    anchorStroke: 'rgb(42,42,42)',
    anchorSize: 12,
    anchorCornerRadius: 3,
    // shiftBehavior: 'none',
    centeredScaling: false,
    anchorStyleFunc: (anchor) => {
      // Make the x/y resize anchors little bars
      if (anchor.hasName('top-center') || anchor.hasName('bottom-center')) {
        anchor.height(8);
        anchor.offsetY(4);
        anchor.width(30);
        anchor.offsetX(15);
      }
      if (anchor.hasName('middle-left') || anchor.hasName('middle-right')) {
        anchor.height(30);
        anchor.offsetY(15);
        anchor.width(8);
        anchor.offsetX(4);
      }
    },
    anchorDragBoundFunc: (oldAbsPos, newAbsPos) => {
      const gridSize = getMetaKey() ? 8 : 64;
      const scaledGridSize = gridSize * stage.scaleX();
      // Calculate the offset of the grid.
      const stageAbsPos = stage.getAbsolutePosition();
      const offsetX = stageAbsPos.x % scaledGridSize;
      const offsetY = stageAbsPos.y % scaledGridSize;
      const finalPos = {
        x: roundToMultiple(newAbsPos.x, scaledGridSize) + offsetX,
        y: roundToMultiple(newAbsPos.y, scaledGridSize) + offsetY,
      };
      console.log('scaledGridSize', scaledGridSize);
      console.log('offsetX', offsetX);
      console.log('offsetY', offsetY);
      console.log('newAbsPosX', newAbsPos.x);
      console.log('newAbsPosY', newAbsPos.y);
      console.log('finalPos', finalPos);
      console.log('finalPosScaled', { x: finalPos.x * stage.scaleX(), y: finalPos.y * stage.scaleY() });

      return finalPos;
    },
  });

  bboxTransformer.on('transform', () => {
    let gridSize = getMetaKey() ? 8 : 64;

    if (getAltKey()) {
      gridSize = gridSize * 2;
    }

    const bbox = {
      x: Math.round(bboxRect.x()),
      y: Math.round(bboxRect.y()),
      width: roundToMultipleMin(bboxRect.width() * bboxRect.scaleX(), gridSize),
      height: roundToMultipleMin(bboxRect.height() * bboxRect.scaleY(), gridSize),
    };
    bboxRect.setAttrs({ ...bbox, scaleX: 1, scaleY: 1 });
    onBboxTransformed(bbox);
  });

  // The transformer will always be transforming the dummy rect
  bboxTransformer.nodes([bboxRect]);
  bboxPreviewGroup.add(bboxRect);
  bboxPreviewGroup.add(bboxTransformer);
  previewLayer.add(bboxPreviewGroup);
  return bboxPreviewGroup;
};

const ALL_ANCHORS: string[] = [
  'top-left',
  'top-center',
  'top-right',
  'middle-right',
  'middle-left',
  'bottom-left',
  'bottom-center',
  'bottom-right',
];
const NO_ANCHORS: string[] = [];

export const renderBboxPreview = (
  stage: Konva.Stage,
  bbox: IRect,
  tool: Tool,
  getBbox: () => IRect,
  onBboxTransformed: (bbox: IRect) => void,
  getShiftKey: () => boolean,
  getMetaKey: () => boolean,
  getAltKey: () => boolean
): void => {
  const bboxGroup = getBboxPreviewGroup(stage, getBbox, onBboxTransformed, getShiftKey, getMetaKey, getAltKey);
  const bboxRect = bboxGroup.findOne<Konva.Rect>(`#${PREVIEW_GENERATION_BBOX_DUMMY_RECT}`);
  const bboxTransformer = bboxGroup.findOne<Konva.Transformer>(`#${PREVIEW_GENERATION_BBOX_TRANSFORMER}`);
  bboxRect?.setAttrs({ ...bbox, listening: tool === 'move' });
  bboxTransformer?.setAttrs({
    listening: tool === 'move',
    enabledAnchors: tool === 'move' ? ALL_ANCHORS : NO_ANCHORS,
  });
};

export const getToolPreviewGroup = (stage: Konva.Stage): Konva.Group => {
  const previewLayer = getPreviewLayer(stage);
  let toolPreviewGroup = previewLayer.findOne<Konva.Group>(`#${PREVIEW_TOOL_GROUP_ID}`);
  if (toolPreviewGroup) {
    return toolPreviewGroup;
  }

  toolPreviewGroup = new Konva.Group({ id: PREVIEW_TOOL_GROUP_ID });

  // Create the brush preview group & circles
  const brushPreviewGroup = new Konva.Group({ id: PREVIEW_BRUSH_GROUP_ID });
  const brushPreviewFill = new Konva.Circle({
    id: PREVIEW_BRUSH_FILL_ID,
    listening: false,
    strokeEnabled: false,
  });
  brushPreviewGroup.add(brushPreviewFill);
  const brushPreviewBorderInner = new Konva.Circle({
    id: PREVIEW_BRUSH_BORDER_INNER_ID,
    listening: false,
    stroke: BRUSH_BORDER_INNER_COLOR,
    strokeWidth: 1,
    strokeEnabled: true,
  });
  brushPreviewGroup.add(brushPreviewBorderInner);
  const brushPreviewBorderOuter = new Konva.Circle({
    id: PREVIEW_BRUSH_BORDER_OUTER_ID,
    listening: false,
    stroke: BRUSH_BORDER_OUTER_COLOR,
    strokeWidth: 1,
    strokeEnabled: true,
  });
  brushPreviewGroup.add(brushPreviewBorderOuter);

  // Create the rect preview - this is a rectangle drawn from the last mouse down position to the current cursor position
  const rectPreview = new Konva.Rect({
    id: PREVIEW_RECT_ID,
    listening: false,
    stroke: BBOX_SELECTED_STROKE,
    strokeWidth: 1,
  });

  toolPreviewGroup.add(rectPreview);
  toolPreviewGroup.add(brushPreviewGroup);
  previewLayer.add(toolPreviewGroup);
  return toolPreviewGroup;
};

/**
 * Renders the preview layer.
 * @param stage The konva stage
 * @param tool The selected tool
 * @param color The selected layer's color
 * @param selectedLayerType The selected layer's type
 * @param globalMaskLayerOpacity The global mask layer opacity
 * @param cursorPos The cursor position
 * @param lastMouseDownPos The position of the last mouse down event - used for the rect tool
 * @param brushSize The brush size
 */
export const renderToolPreview = (
  stage: Konva.Stage,
  tool: Tool,
  brushColor: RgbaColor,
  selectedLayerType: Layer['type'] | null,
  globalMaskLayerOpacity: number,
  cursorPos: Vector2d | null,
  lastMouseDownPos: Vector2d | null,
  brushSize: number,
  isDrawing: boolean,
  isMouseDown: boolean
): void => {
  const layerCount = stage.find(selectRenderableLayers).length;
  // Update the stage's pointer style
  if (tool === 'view') {
    // View gets a hand
    stage.container().style.cursor = isMouseDown ? 'grabbing' : 'grab';
  } else if (layerCount === 0) {
    // We have no layers, so we should not render any tool
    stage.container().style.cursor = 'default';
  } else if (selectedLayerType !== 'regional_guidance_layer' && selectedLayerType !== 'raster_layer') {
    // Non-mask-guidance layers don't have tools
    stage.container().style.cursor = 'not-allowed';
  } else if (tool === 'move') {
    // Move tool gets a pointer
    stage.container().style.cursor = 'default';
  } else if (tool === 'rect') {
    // Rect gets a crosshair
    stage.container().style.cursor = 'crosshair';
  } else {
    // Else we hide the native cursor and use the konva-rendered brush preview
    stage.container().style.cursor = 'none';
  }

  stage.draggable(tool === 'view');

  const toolPreviewGroup = getToolPreviewGroup(stage);

  if (
    !cursorPos ||
    layerCount === 0 ||
    (selectedLayerType !== 'regional_guidance_layer' && selectedLayerType !== 'raster_layer')
  ) {
    // We can bail early if the mouse isn't over the stage or there are no layers
    toolPreviewGroup.visible(false);
  } else {
    toolPreviewGroup.visible(true);

    const brushPreviewGroup = stage.findOne<Konva.Group>(`#${PREVIEW_BRUSH_GROUP_ID}`);
    assert(brushPreviewGroup, 'Brush preview group not found');

    const rectPreview = stage.findOne<Konva.Rect>(`#${PREVIEW_RECT_ID}`);
    assert(rectPreview, 'Rect preview not found');

    // No need to render the brush preview if the cursor position or color is missing
    if (cursorPos && (tool === 'brush' || tool === 'eraser')) {
      // Update the fill circle
      const brushPreviewFill = brushPreviewGroup.findOne<Konva.Circle>(`#${PREVIEW_BRUSH_FILL_ID}`);
      brushPreviewFill?.setAttrs({
        x: cursorPos.x,
        y: cursorPos.y,
        radius: brushSize / 2,
        fill: isDrawing ? '' : rgbaColorToString(brushColor),
        globalCompositeOperation: tool === 'brush' ? 'source-over' : 'destination-out',
      });

      // Update the inner border of the brush preview
      const brushPreviewInner = brushPreviewGroup.findOne<Konva.Circle>(`#${PREVIEW_BRUSH_BORDER_INNER_ID}`);
      brushPreviewInner?.setAttrs({ x: cursorPos.x, y: cursorPos.y, radius: brushSize / 2 });

      // Update the outer border of the brush preview
      const brushPreviewOuter = brushPreviewGroup.findOne<Konva.Circle>(`#${PREVIEW_BRUSH_BORDER_OUTER_ID}`);
      brushPreviewOuter?.setAttrs({
        x: cursorPos.x,
        y: cursorPos.y,
        radius: brushSize / 2 + 1,
      });

      brushPreviewGroup.visible(true);
    } else {
      brushPreviewGroup.visible(false);
    }

    if (cursorPos && lastMouseDownPos && tool === 'rect') {
      const snappedPos = snapPosToStage(cursorPos, stage);
      const rectPreview = brushPreviewGroup.findOne<Konva.Rect>(`#${PREVIEW_RECT_ID}`);
      rectPreview?.setAttrs({
        x: Math.min(snappedPos.x, lastMouseDownPos.x),
        y: Math.min(snappedPos.y, lastMouseDownPos.y),
        width: Math.abs(snappedPos.x - lastMouseDownPos.x),
        height: Math.abs(snappedPos.y - lastMouseDownPos.y),
        fill: rgbaColorToString(brushColor),
      });
      rectPreview?.visible(true);
    } else {
      rectPreview?.visible(false);
    }
  }
};
