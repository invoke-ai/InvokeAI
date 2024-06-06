import { rgbaColorToString } from 'features/canvas/util/colorToString';
import {
  BBOX_SELECTED_STROKE,
  BRUSH_BORDER_INNER_COLOR,
  BRUSH_BORDER_OUTER_COLOR,
} from 'features/controlLayers/konva/constants';
import {
  TOOL_PREVIEW_BRUSH_BORDER_INNER_ID,
  TOOL_PREVIEW_BRUSH_BORDER_OUTER_ID,
  TOOL_PREVIEW_BRUSH_FILL_ID,
  TOOL_PREVIEW_BRUSH_GROUP_ID,
  TOOL_PREVIEW_LAYER_ID,
  TOOL_PREVIEW_RECT_ID,
} from 'features/controlLayers/konva/naming';
import { selectRenderableLayers, snapPosToStage } from 'features/controlLayers/konva/util';
import type { Layer, RgbaColor, Tool } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import { assert } from 'tsafe';

/**
 * Logic to create and render the singleton tool preview layer.
 */

/**
 * Creates the singleton tool preview layer and all its objects.
 * @param stage The konva stage
 */
const createToolPreviewLayer = (stage: Konva.Stage): Konva.Layer => {
  // Initialize the brush preview layer & add to the stage
  const toolPreviewLayer = new Konva.Layer({ id: TOOL_PREVIEW_LAYER_ID, visible: false, listening: false });
  stage.add(toolPreviewLayer);

  // Create the brush preview group & circles
  const brushPreviewGroup = new Konva.Group({ id: TOOL_PREVIEW_BRUSH_GROUP_ID });
  const brushPreviewFill = new Konva.Circle({
    id: TOOL_PREVIEW_BRUSH_FILL_ID,
    listening: false,
    strokeEnabled: false,
  });
  brushPreviewGroup.add(brushPreviewFill);
  const brushPreviewBorderInner = new Konva.Circle({
    id: TOOL_PREVIEW_BRUSH_BORDER_INNER_ID,
    listening: false,
    stroke: BRUSH_BORDER_INNER_COLOR,
    strokeWidth: 1,
    strokeEnabled: true,
  });
  brushPreviewGroup.add(brushPreviewBorderInner);
  const brushPreviewBorderOuter = new Konva.Circle({
    id: TOOL_PREVIEW_BRUSH_BORDER_OUTER_ID,
    listening: false,
    stroke: BRUSH_BORDER_OUTER_COLOR,
    strokeWidth: 1,
    strokeEnabled: true,
  });
  brushPreviewGroup.add(brushPreviewBorderOuter);
  toolPreviewLayer.add(brushPreviewGroup);

  // Create the rect preview - this is a rectangle drawn from the last mouse down position to the current cursor position
  const rectPreview = new Konva.Rect({
    id: TOOL_PREVIEW_RECT_ID,
    listening: false,
    stroke: BBOX_SELECTED_STROKE,
    strokeWidth: 1,
  });
  toolPreviewLayer.add(rectPreview);

  return toolPreviewLayer;
};

/**
 * Renders the brush preview for the selected tool.
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
  isDrawing: boolean
): void => {
  const layerCount = stage.find(selectRenderableLayers).length;
  // Update the stage's pointer style
  if (layerCount === 0) {
    // We have no layers, so we should not render any tool
    stage.container().style.cursor = 'default';
  } else if (selectedLayerType !== 'regional_guidance_layer' && selectedLayerType !== 'raster_layer') {
    // Non-mask-guidance layers don't have tools
    stage.container().style.cursor = 'not-allowed';
  } else if (tool === 'move') {
    // Move tool gets a pointer
    stage.container().style.cursor = 'default';
  } else if (tool === 'rect') {
    // Move rect gets a crosshair
    stage.container().style.cursor = 'crosshair';
  } else {
    // Else we hide the native cursor and use the konva-rendered brush preview
    stage.container().style.cursor = 'none';
  }

  const toolPreviewLayer = stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`) ?? createToolPreviewLayer(stage);

  if (!cursorPos || layerCount === 0) {
    // We can bail early if the mouse isn't over the stage or there are no layers
    toolPreviewLayer.visible(false);
    return;
  }

  toolPreviewLayer.visible(true);

  const brushPreviewGroup = stage.findOne<Konva.Group>(`#${TOOL_PREVIEW_BRUSH_GROUP_ID}`);
  assert(brushPreviewGroup, 'Brush preview group not found');

  const rectPreview = stage.findOne<Konva.Rect>(`#${TOOL_PREVIEW_RECT_ID}`);
  assert(rectPreview, 'Rect preview not found');

  // No need to render the brush preview if the cursor position or color is missing
  if (cursorPos && (tool === 'brush' || tool === 'eraser')) {
    // Update the fill circle
    const brushPreviewFill = brushPreviewGroup.findOne<Konva.Circle>(`#${TOOL_PREVIEW_BRUSH_FILL_ID}`);
    brushPreviewFill?.setAttrs({
      x: cursorPos.x,
      y: cursorPos.y,
      radius: brushSize / 2,
      fill: isDrawing ? '' : rgbaColorToString(brushColor),
      globalCompositeOperation: tool === 'brush' ? 'source-over' : 'destination-out',
    });

    // Update the inner border of the brush preview
    const brushPreviewInner = toolPreviewLayer.findOne<Konva.Circle>(`#${TOOL_PREVIEW_BRUSH_BORDER_INNER_ID}`);
    brushPreviewInner?.setAttrs({ x: cursorPos.x, y: cursorPos.y, radius: brushSize / 2 });

    // Update the outer border of the brush preview
    const brushPreviewOuter = toolPreviewLayer.findOne<Konva.Circle>(`#${TOOL_PREVIEW_BRUSH_BORDER_OUTER_ID}`);
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
    const rectPreview = toolPreviewLayer.findOne<Konva.Rect>(`#${TOOL_PREVIEW_RECT_ID}`);
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
};
