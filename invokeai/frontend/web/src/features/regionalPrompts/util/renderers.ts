import { getStore } from 'app/store/nanostores/store';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import { getScaledFlooredCursorPosition } from 'features/regionalPrompts/hooks/mouseEventHooks';
import type { Layer, RegionalPromptLayer, RPTool } from 'features/regionalPrompts/store/regionalPromptsSlice';
import {
  $isMouseOver,
  $tool,
  BRUSH_PREVIEW_BORDER_INNER_ID,
  BRUSH_PREVIEW_BORDER_OUTER_ID,
  BRUSH_PREVIEW_FILL_ID,
  BRUSH_PREVIEW_LAYER_ID,
  getPRLayerBboxId,
  getRPLayerObjectGroupId,
  REGIONAL_PROMPT_LAYER_BBOX_NAME,
  REGIONAL_PROMPT_LAYER_LINE_NAME,
  REGIONAL_PROMPT_LAYER_NAME,
  REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getKonvaLayerBbox } from 'features/regionalPrompts/util/bbox';
import Konva from 'konva';
import type { IRect, Vector2d } from 'konva/lib/types';
import type { RgbColor } from 'react-colorful';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

const BBOX_SELECTED_STROKE = 'rgba(78, 190, 255, 1)';
const BBOX_NOT_SELECTED_STROKE = 'rgba(255, 255, 255, 0.353)';
const BBOX_NOT_SELECTED_MOUSEOVER_STROKE = 'rgba(255, 255, 255, 0.661)';
const BRUSH_PREVIEW_BORDER_INNER_COLOR = 'rgba(0,0,0,1)';
const BRUSH_PREVIEW_BORDER_OUTER_COLOR = 'rgba(255,255,255,0.8)';
const GET_CLIENT_RECT_CONFIG = { skipTransform: true };

const mapId = (object: { id: string }) => object.id;
const getIsSelected = (layerId?: string | null) => {
  if (!layerId) {
    return false;
  }
  return layerId === getStore().getState().regionalPrompts.present.selectedLayerId;
};

/**
 * Renders the brush preview for the selected tool.
 * @param stage The konva stage to render on.
 * @param tool The selected tool.
 * @param color The selected layer's color.
 * @param cursorPos The cursor position.
 * @param brushSize The brush size.
 */
export const renderBrushPreview = (
  stage: Konva.Stage,
  tool: RPTool,
  color: RgbColor | null,
  cursorPos: Vector2d | null,
  brushSize: number
) => {
  const layerCount = stage.find(`.${REGIONAL_PROMPT_LAYER_NAME}`).length;
  // Update the stage's pointer style
  stage.container().style.cursor = tool === 'move' || layerCount === 0 ? 'default' : 'none';

  // Create the layer if it doesn't exist
  let layer = stage.findOne<Konva.Layer>(`#${BRUSH_PREVIEW_LAYER_ID}`);
  if (!layer) {
    // Initialize the brush preview layer & add to the stage
    layer = new Konva.Layer({ id: BRUSH_PREVIEW_LAYER_ID, visible: tool !== 'move', listening: false });
    stage.add(layer);
    // The brush preview is hidden and shown as the mouse leaves and enters the stage
    stage.on('mousemove', (e) => {
      e.target
        .getStage()
        ?.findOne<Konva.Layer>(`#${BRUSH_PREVIEW_LAYER_ID}`)
        ?.visible($tool.get() !== 'move');
    });
    stage.on('mouseleave', (e) => {
      e.target.getStage()?.findOne<Konva.Layer>(`#${BRUSH_PREVIEW_LAYER_ID}`)?.visible(false);
    });
    stage.on('mouseenter', (e) => {
      e.target
        .getStage()
        ?.findOne<Konva.Layer>(`#${BRUSH_PREVIEW_LAYER_ID}`)
        ?.visible($tool.get() !== 'move');
    });
  }

  if (!$isMouseOver.get()) {
    layer.visible(false);
    return;
  }

  // ...but we may want to hide it if it is visible, when using the move tool or when there are no layers
  layer.visible(tool !== 'move' && layerCount > 0);

  // No need to render the brush preview if the cursor position or color is missing
  if (!cursorPos || !color) {
    return;
  }

  // Create and/or update the fill circle
  let fill = layer.findOne<Konva.Circle>(`#${BRUSH_PREVIEW_FILL_ID}`);
  if (!fill) {
    fill = new Konva.Circle({
      id: BRUSH_PREVIEW_FILL_ID,
      listening: false,
      strokeEnabled: false,
    });
    layer.add(fill);
  }
  fill.setAttrs({
    x: cursorPos.x,
    y: cursorPos.y,
    radius: brushSize / 2,
    fill: rgbColorToString(color),
    globalCompositeOperation: tool === 'brush' ? 'source-over' : 'destination-out',
  });

  // Create and/or update the inner border of the brush preview
  let borderInner = layer.findOne<Konva.Circle>(`#${BRUSH_PREVIEW_BORDER_INNER_ID}`);
  if (!borderInner) {
    borderInner = new Konva.Circle({
      id: BRUSH_PREVIEW_BORDER_INNER_ID,
      listening: false,
      stroke: BRUSH_PREVIEW_BORDER_INNER_COLOR,
      strokeWidth: 1,
      strokeEnabled: true,
    });
    layer.add(borderInner);
  }
  borderInner.setAttrs({ x: cursorPos.x, y: cursorPos.y, radius: brushSize / 2 });

  // Create and/or update the outer border of the brush preview
  let borderOuter = layer.findOne<Konva.Circle>(`#${BRUSH_PREVIEW_BORDER_OUTER_ID}`);
  if (!borderOuter) {
    borderOuter = new Konva.Circle({
      id: BRUSH_PREVIEW_BORDER_OUTER_ID,
      listening: false,
      stroke: BRUSH_PREVIEW_BORDER_OUTER_COLOR,
      strokeWidth: 1,
      strokeEnabled: true,
    });
    layer.add(borderOuter);
  }
  borderOuter.setAttrs({
    x: cursorPos.x,
    y: cursorPos.y,
    radius: brushSize / 2 + 1,
  });
};

const renderRPLayer = (
  stage: Konva.Stage,
  rpLayer: RegionalPromptLayer,
  rpLayerIndex: number,
  selectedLayerIdId: string | null,
  tool: RPTool,
  layerOpacity: number,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
) => {
  let konvaLayer = stage.findOne<Konva.Layer>(`#${rpLayer.id}`);

  if (!konvaLayer) {
    // This layer hasn't been added to the konva state yet
    konvaLayer = new Konva.Layer({
      id: rpLayer.id,
      name: REGIONAL_PROMPT_LAYER_NAME,
      draggable: true,
      dragDistance: 0,
    });

    // Create a `dragmove` listener for this layer
    if (onLayerPosChanged) {
      konvaLayer.on('dragend', function (e) {
        onLayerPosChanged(rpLayer.id, Math.floor(e.target.x()), Math.floor(e.target.y()));
      });
    }

    // The dragBoundFunc limits how far the layer can be dragged
    konvaLayer.dragBoundFunc(function (pos) {
      const cursorPos = getScaledFlooredCursorPosition(stage);
      if (!cursorPos) {
        return this.getAbsolutePosition();
      }
      // Prevent the user from dragging the layer out of the stage bounds.
      if (
        cursorPos.x < 0 ||
        cursorPos.x > stage.width() / stage.scaleX() ||
        cursorPos.y < 0 ||
        cursorPos.y > stage.height() / stage.scaleY()
      ) {
        return this.getAbsolutePosition();
      }
      return pos;
    });

    // The object group holds all of the layer's objects (e.g. lines and rects)
    const konvaObjectGroup = new Konva.Group({
      id: getRPLayerObjectGroupId(rpLayer.id, uuidv4()),
      name: REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME,
      listening: false,
    });
    konvaLayer.add(konvaObjectGroup);

    stage.add(konvaLayer);

    // When a layer is added, it ends up on top of the brush preview - we need to move the preview back to the top.
    stage.findOne<Konva.Layer>(`#${BRUSH_PREVIEW_LAYER_ID}`)?.moveToTop();
  }

  // Update the layer's position and listening state
  konvaLayer.setAttrs({
    listening: tool === 'move', // The layer only listens when using the move tool - otherwise the stage is handling mouse events
    x: Math.floor(rpLayer.x),
    y: Math.floor(rpLayer.y),
    // There are rpLayers.length layers, plus a brush preview layer rendered on top of them, so the zIndex works
    // out to be the layerIndex. If more layers are added, this may no longer be true.
    zIndex: rpLayerIndex,
  });

  const color = rgbColorToString(rpLayer.color);

  const konvaObjectGroup = konvaLayer.findOne<Konva.Group>(`.${REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME}`);
  assert(konvaObjectGroup, `Object group not found for layer ${rpLayer.id}`);

  // We use caching to handle "global" layer opacity, but caching is expensive and we should only do it when required.
  let groupNeedsCache = false;

  if (konvaObjectGroup.opacity() !== layerOpacity) {
    konvaObjectGroup.opacity(layerOpacity);
  }

  // Remove deleted objects
  const objectIds = rpLayer.objects.map(mapId);
  for (const objectNode of konvaLayer.find(`.${REGIONAL_PROMPT_LAYER_LINE_NAME}`)) {
    if (!objectIds.includes(objectNode.id())) {
      objectNode.destroy();
      groupNeedsCache = true;
    }
  }

  for (const reduxObject of rpLayer.objects) {
    // TODO: Handle rects, images, etc
    if (reduxObject.kind !== 'line') {
      continue;
    }

    let konvaObject = stage.findOne<Konva.Line>(`#${reduxObject.id}`);

    if (!konvaObject) {
      // This object hasn't been added to the konva state yet.
      konvaObject = new Konva.Line({
        id: reduxObject.id,
        key: reduxObject.id,
        name: REGIONAL_PROMPT_LAYER_LINE_NAME,
        strokeWidth: reduxObject.strokeWidth,
        tension: 0,
        lineCap: 'round',
        lineJoin: 'round',
        shadowForStrokeEnabled: false,
        globalCompositeOperation: reduxObject.tool === 'brush' ? 'source-over' : 'destination-out',
        listening: false,
      });
      konvaObjectGroup.add(konvaObject);
    }

    // Only update the points if they have changed. The point values are never mutated, they are only added to the array.
    if (konvaObject.points().length !== reduxObject.points.length) {
      konvaObject.points(reduxObject.points);
      groupNeedsCache = true;
    }
    // Only update the color if it has changed.
    if (konvaObject.stroke() !== color) {
      konvaObject.stroke(color);
      groupNeedsCache = true;
    }
    // Only update layer visibility if it has changed.
    if (konvaLayer.visible() !== rpLayer.isVisible) {
      konvaLayer.visible(rpLayer.isVisible);
      groupNeedsCache = true;
    }
  }

  if (groupNeedsCache) {
    konvaObjectGroup.cache();
  }
};

/**
 * Renders the layers on the stage.
 * @param stage The konva stage to render on.
 * @param reduxLayers Array of the layers from the redux store.
 * @param selectedLayerIdId The selected layer id.
 * @param layerOpacity The opacity of the layer.
 * @param onLayerPosChanged Callback for when the layer's position changes. This is optional to allow for offscreen rendering.
 * @returns
 */
export const renderLayers = (
  stage: Konva.Stage,
  reduxLayers: Layer[],
  selectedLayerIdId: string | null,
  layerOpacity: number,
  tool: RPTool,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
) => {
  const reduxLayerIds = reduxLayers.map(mapId);

  // Remove un-rendered layers
  for (const konvaLayer of stage.find<Konva.Layer>(`.${REGIONAL_PROMPT_LAYER_NAME}`)) {
    if (!reduxLayerIds.includes(konvaLayer.id())) {
      konvaLayer.destroy();
    }
  }

  for (let layerIndex = 0; layerIndex < reduxLayers.length; layerIndex++) {
    const reduxLayer = reduxLayers[layerIndex];
    assert(reduxLayer, `Layer at index ${layerIndex} is undefined`);
    if (reduxLayer.kind === 'regionalPromptLayer') {
      renderRPLayer(stage, reduxLayer, layerIndex, selectedLayerIdId, tool, layerOpacity, onLayerPosChanged);
    }
  }
};

/**
 *
 * @param stage The konva stage to render on.
 * @param tool The current tool.
 * @param selectedLayerIdId The currently selected layer id.
 * @param onBboxChanged A callback to be called when the bounding box changes.
 * @returns
 */
export const renderBbox = (
  stage: Konva.Stage,
  reduxLayers: Layer[],
  selectedLayerId: string | null,
  tool: RPTool,
  onBboxChanged: (layerId: string, bbox: IRect | null) => void,
  onBboxMouseDown: (layerId: string) => void
) => {
  // No selected layer or not using the move tool - nothing more to do here
  if (tool !== 'move') {
    for (const bboxRect of stage.find<Konva.Rect>(`.${REGIONAL_PROMPT_LAYER_BBOX_NAME}`)) {
      bboxRect.visible(false);
      bboxRect.listening(false);
    }
    return;
  }

  for (const reduxLayer of reduxLayers) {
    const konvaLayer = stage.findOne<Konva.Layer>(`#${reduxLayer.id}`);
    assert(konvaLayer, `Layer ${reduxLayer.id} not found in stage`);

    let bbox = reduxLayer.bbox;

    // We only need to recalculate the bbox if the layer has changed and it has objects
    if (reduxLayer.bboxNeedsUpdate && reduxLayer.objects.length) {
      // We only need to use the pixel-perfect bounding box if the layer has eraser strokes
      bbox = reduxLayer.hasEraserStrokes
        ? getKonvaLayerBbox(konvaLayer)
        : konvaLayer.getClientRect(GET_CLIENT_RECT_CONFIG);

      // Update the layer's bbox in the redux store
      onBboxChanged(reduxLayer.id, bbox);
    }

    if (!bbox) {
      continue;
    }

    let rect = konvaLayer.findOne<Konva.Rect>(`.${REGIONAL_PROMPT_LAYER_BBOX_NAME}`);
    if (!rect) {
      rect = new Konva.Rect({
        id: getPRLayerBboxId(reduxLayer.id),
        name: REGIONAL_PROMPT_LAYER_BBOX_NAME,
        strokeWidth: 1,
      });
      rect.on('mousedown', function () {
        onBboxMouseDown(reduxLayer.id);
      });
      rect.on('mouseover', function (e) {
        if (getIsSelected(e.target.getLayer()?.id())) {
          this.stroke(BBOX_SELECTED_STROKE);
        } else {
          this.stroke(BBOX_NOT_SELECTED_MOUSEOVER_STROKE);
        }
      });
      rect.on('mouseout', function (e) {
        if (getIsSelected(e.target.getLayer()?.id())) {
          this.stroke(BBOX_SELECTED_STROKE);
        } else {
          this.stroke(BBOX_NOT_SELECTED_STROKE);
        }
      });
      konvaLayer.add(rect);
    }

    rect.setAttrs({
      visible: true,
      listening: true,
      x: bbox.x - 1,
      y: bbox.y - 1,
      width: bbox.width + 2,
      height: bbox.height + 2,
      stroke: reduxLayer.id === selectedLayerId ? BBOX_SELECTED_STROKE : BBOX_NOT_SELECTED_STROKE,
    });
  }
};
