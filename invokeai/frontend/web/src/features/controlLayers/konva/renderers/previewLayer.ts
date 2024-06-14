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
import { selectRenderableLayers } from 'features/controlLayers/konva/util';
import type { CanvasEntity, CanvasV2State, RgbaColor, Tool } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { IRect, Vector2d } from 'konva/lib/types';
import { atom } from 'nanostores';
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
  getCtrlKey: () => boolean,
  getMetaKey: () => boolean,
  getAltKey: () => boolean
): Konva.Group => {
  const previewLayer = getPreviewLayer(stage);
  let bboxPreviewGroup = previewLayer.findOne<Konva.Group>(`#${PREVIEW_GENERATION_BBOX_GROUP}`);

  if (bboxPreviewGroup) {
    return bboxPreviewGroup;
  }

  // Create a stash to hold onto the last aspect ratio of the bbox - this allows for locking the aspect ratio when
  // transforming the bbox.
  const bbox = getBbox();
  const $aspectRatioBuffer = atom(bbox.width / bbox.height);

  // Use a transformer for the generation bbox. Transformers need some shape to transform, we will use a fully
  // transparent rect for this purpose.
  bboxPreviewGroup = new Konva.Group({ id: PREVIEW_GENERATION_BBOX_GROUP, listening: false });
  const bboxRect = new Konva.Rect({
    id: PREVIEW_GENERATION_BBOX_DUMMY_RECT,
    listening: false,
    strokeEnabled: false,
    draggable: true,
    ...getBbox(),
  });
  bboxRect.on('dragmove', () => {
    const gridSize = getCtrlKey() || getMetaKey() ? 8 : 64;
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
    shiftBehavior: 'none', // we will implement our own shift behavior
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
    anchorDragBoundFunc: (_oldAbsPos, newAbsPos) => {
      // This function works with absolute position - that is, a position in "physical" pixels on the screen, as opposed
      // to konva's internal coordinate system.

      // We need to snap the anchors to the grid. If the user is holding ctrl/meta, we use the finer 8px grid.
      const gridSize = getCtrlKey() || getMetaKey() ? 8 : 64;
      // Because we are working in absolute coordinates, we need to scale the grid size by the stage scale.
      const scaledGridSize = gridSize * stage.scaleX();
      // To snap the anchor to the grid, we need to calculate an offset from the stage's absolute position.
      const stageAbsPos = stage.getAbsolutePosition();
      // The offset is the remainder of the stage's absolute position divided by the scaled grid size.
      const offsetX = stageAbsPos.x % scaledGridSize;
      const offsetY = stageAbsPos.y % scaledGridSize;
      // Finally, calculate the position by rounding to the grid and adding the offset.
      return {
        x: roundToMultiple(newAbsPos.x, scaledGridSize) + offsetX,
        y: roundToMultiple(newAbsPos.y, scaledGridSize) + offsetY,
      };
    },
  });

  bboxTransformer.on('transform', () => {
    // In the transform callback, we calculate the bbox's new dims and pos and update the konva object.

    // Some special handling is needed depending on the anchor being dragged.
    const anchor = bboxTransformer.getActiveAnchor();
    if (!anchor) {
      // Pretty sure we should always have an anchor here?
      return;
    }

    const alt = getAltKey();
    const ctrl = getCtrlKey();
    const meta = getMetaKey();
    const shift = getShiftKey();

    // Grid size depends on the modifier keys
    let gridSize = ctrl || meta ? 8 : 64;

    // Alt key indicates we are using centered scaling. We need to double the gride size used when calculating the
    // new dimensions so that each size scales in the correct increments and doesn't mis-place the bbox. For example, if
    // we snapped the width and height to 8px increments, the bbox would be mis-placed by 4px in the x and y axes.
    // Doubling the grid size ensures the bbox's coords remain aligned to the 8px/64px grid.
    if (getAltKey()) {
      gridSize = gridSize * 2;
    }

    // The coords should be correct per the anchorDragBoundFunc.
    let x = bboxRect.x();
    let y = bboxRect.y();

    // Konva transforms by scaling the dims, not directly changing width and height. At this point, the width and height
    // *have not changed*, only the scale has changed. To get the final height, we need to scale the dims and then snap
    // them to the grid.
    let width = roundToMultipleMin(bboxRect.width() * bboxRect.scaleX(), gridSize);
    let height = roundToMultipleMin(bboxRect.height() * bboxRect.scaleY(), gridSize);

    // If shift is held and we are resizing from a corner, retain aspect ratio - needs special handling. We skip this
    // if alt/opt is held - this requires math too big for my brain.
    if (shift && CORNER_ANCHORS.includes(anchor) && !alt) {
      // Fit the bbox to the last aspect ratio
      let fittedWidth = Math.sqrt(width * height * $aspectRatioBuffer.get());
      let fittedHeight = fittedWidth / $aspectRatioBuffer.get();
      fittedWidth = roundToMultipleMin(fittedWidth, gridSize);
      fittedHeight = roundToMultipleMin(fittedHeight, gridSize);

      // We need to adjust the x and y coords to have the resize occur from the right origin.
      if (anchor === 'top-left') {
        // The transform origin is the bottom-right anchor. Both x and y need to be updated.
        x = x - (fittedWidth - width);
        y = y - (fittedHeight - height);
      }
      if (anchor === 'top-right') {
        // The transform origin is the bottom-left anchor. Only y needs to be updated.
        y = y - (fittedHeight - height);
      }
      if (anchor === 'bottom-left') {
        // The transform origin is the top-right anchor. Only x needs to be updated.
        x = x - (fittedWidth - width);
      }
      // Update the width and height to the fitted dims.
      width = fittedWidth;
      height = fittedHeight;
    }

    const bbox = {
      x: Math.round(x),
      y: Math.round(y),
      width,
      height,
    };

    // Here we _could_ go ahead and update the bboxRect's attrs directly with the new transform, and reset its scale to 1.
    // However, we have another function that renders the bbox when its internal state changes, so we will rely on that
    // to set the new attrs.

    // Update the bbox in internal state.
    onBboxTransformed(bbox);

    // Update the aspect ratio buffer whenever the shift key is not held - this allows for a nice UX where you can start
    // a transform, get the right aspect ratio, then hold shift to lock it in.
    if (!shift) {
      $aspectRatioBuffer.set(bbox.width / bbox.height);
    }
  });

  bboxTransformer.on('transformend', () => {
    // Always update the aspect ratio buffer when the transform ends, so if the next transform starts with shift held,
    // we have the correct aspect ratio to start from.
    $aspectRatioBuffer.set(bboxRect.width() / bboxRect.height());
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
const CORNER_ANCHORS: string[] = ['top-left', 'top-right', 'bottom-left', 'bottom-right'];
const NO_ANCHORS: string[] = [];

export const renderBboxPreview = (
  stage: Konva.Stage,
  bbox: IRect,
  tool: Tool,
  getBbox: () => IRect,
  onBboxTransformed: (bbox: IRect) => void,
  getShiftKey: () => boolean,
  getCtrlKey: () => boolean,
  getMetaKey: () => boolean,
  getAltKey: () => boolean
): void => {
  const bboxGroup = getBboxPreviewGroup(
    stage,
    getBbox,
    onBboxTransformed,
    getShiftKey,
    getCtrlKey,
    getMetaKey,
    getAltKey
  );
  const bboxRect = bboxGroup.findOne<Konva.Rect>(`#${PREVIEW_GENERATION_BBOX_DUMMY_RECT}`);
  const bboxTransformer = bboxGroup.findOne<Konva.Transformer>(`#${PREVIEW_GENERATION_BBOX_TRANSFORMER}`);
  bboxGroup.listening(tool === 'bbox');
  // This updates the bbox during transformation
  bboxRect?.setAttrs({ ...bbox, scaleX: 1, scaleY: 1, listening: tool === 'bbox' });
  bboxTransformer?.setAttrs({ listening: tool === 'bbox', enabledAnchors: tool === 'bbox' ? ALL_ANCHORS : NO_ANCHORS });
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
 * @param currentFill The selected layer's color
 * @param selectedEntity The selected layer's type
 * @param globalMaskLayerOpacity The global mask layer opacity
 * @param cursorPos The cursor position
 * @param lastMouseDownPos The position of the last mouse down event - used for the rect tool
 * @param brushSize The brush size
 */
export const renderToolPreview = (
  stage: Konva.Stage,
  toolState: CanvasV2State['tool'],
  currentFill: RgbaColor,
  selectedEntity: CanvasEntity | null,
  cursorPos: Vector2d | null,
  lastMouseDownPos: Vector2d | null,
  isDrawing: boolean,
  isMouseDown: boolean
): void => {
  const layerCount = stage.find(selectRenderableLayers).length;
  const tool = toolState.selected;
  // Update the stage's pointer style
  if (tool === 'view') {
    // View gets a hand
    stage.container().style.cursor = isMouseDown ? 'grabbing' : 'grab';
  } else if (layerCount === 0) {
    // We have no layers, so we should not render any tool
    stage.container().style.cursor = 'default';
  } else if (selectedEntity?.type !== 'regional_guidance' && selectedEntity?.type !== 'layer') {
    // Non-mask-guidance layers don't have tools
    stage.container().style.cursor = 'not-allowed';
  } else if (tool === 'move') {
    // Move tool gets a pointer
    stage.container().style.cursor = 'default';
  } else if (tool === 'rect') {
    // Rect gets a crosshair
    stage.container().style.cursor = 'crosshair';
  } else if (tool === 'brush' || tool === 'eraser') {
    // Hide the native cursor and use the konva-rendered brush preview
    stage.container().style.cursor = 'none';
  } else if (tool === 'bbox') {
    stage.container().style.cursor = 'default';
  }

  stage.draggable(tool === 'view');

  const toolPreviewGroup = getToolPreviewGroup(stage);

  if (
    !cursorPos ||
    layerCount === 0 ||
    (selectedEntity?.type !== 'regional_guidance' && selectedEntity?.type !== 'layer')
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
      const radius = (tool === 'brush' ? toolState.brush.width : toolState.eraser.width) / 2;
      brushPreviewFill?.setAttrs({
        x: cursorPos.x,
        y: cursorPos.y,
        radius,
        fill: isDrawing ? '' : rgbaColorToString(currentFill),
        globalCompositeOperation: tool === 'brush' ? 'source-over' : 'destination-out',
      });

      // Update the inner border of the brush preview
      const brushPreviewInner = brushPreviewGroup.findOne<Konva.Circle>(`#${PREVIEW_BRUSH_BORDER_INNER_ID}`);
      brushPreviewInner?.setAttrs({ x: cursorPos.x, y: cursorPos.y, radius });

      // Update the outer border of the brush preview
      const brushPreviewOuter = brushPreviewGroup.findOne<Konva.Circle>(`#${PREVIEW_BRUSH_BORDER_OUTER_ID}`);
      brushPreviewOuter?.setAttrs({
        x: cursorPos.x,
        y: cursorPos.y,
        radius: radius + 1,
      });

      brushPreviewGroup.visible(true);
    } else {
      brushPreviewGroup.visible(false);
    }

    if (cursorPos && lastMouseDownPos && tool === 'rect') {
      const rectPreview = toolPreviewGroup.findOne<Konva.Rect>(`#${PREVIEW_RECT_ID}`);
      rectPreview?.setAttrs({
        x: Math.min(cursorPos.x, lastMouseDownPos.x),
        y: Math.min(cursorPos.y, lastMouseDownPos.y),
        width: Math.abs(cursorPos.x - lastMouseDownPos.x),
        height: Math.abs(cursorPos.y - lastMouseDownPos.y),
        fill: rgbaColorToString(currentFill),
      });
      rectPreview?.visible(true);
    } else {
      rectPreview?.visible(false);
    }
  }
};
