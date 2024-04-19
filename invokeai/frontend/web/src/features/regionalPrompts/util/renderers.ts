import { rgbColorToString } from 'features/canvas/util/colorToString';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import type { Layer, RegionalPromptLayer, RPTool } from 'features/regionalPrompts/store/regionalPromptsSlice';
import {
  BRUSH_PREVIEW_BORDER_INNER_ID,
  BRUSH_PREVIEW_BORDER_OUTER_ID,
  BRUSH_PREVIEW_FILL_ID,
  BRUSH_PREVIEW_LAYER_ID,
  getPRLayerBboxId,
  getRPLayerObjectGroupId,
  getRPLayerTransparencyRectId,
  REGIONAL_PROMPT_LAYER_BBOX_NAME,
  REGIONAL_PROMPT_LAYER_LINE_NAME,
  REGIONAL_PROMPT_LAYER_NAME,
  REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getKonvaLayerBbox } from 'features/regionalPrompts/util/bbox';
import Konva from 'konva';
import type { Node, NodeConfig } from 'konva/lib/Node';
import type { IRect, Vector2d } from 'konva/lib/types';
import type { RgbColor } from 'react-colorful';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

const BRUSH_PREVIEW_BORDER_INNER_COLOR = 'rgba(0,0,0,1)';
const BRUSH_PREVIEW_BORDER_OUTER_COLOR = 'rgba(255,255,255,0.8)';
const mapId = (object: { id: string }) => object.id;

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
    stage.on('mouseleave', (e) => {
      e.target.getStage()?.findOne<Konva.Layer>(`#${BRUSH_PREVIEW_LAYER_ID}`)?.visible(false);
    });
    stage.on('mouseenter', (e) => {
      e.target.getStage()?.findOne<Konva.Layer>(`#${BRUSH_PREVIEW_LAYER_ID}`)?.visible(true);
    });
  }

  if (!layer.visible()) {
    // Rely on the mouseenter and mouseleave events as a "first pass" for brush preview visibility. If it is not visible
    // inside this render function, we do not want to make it visible again...
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
    });

    // Create a `dragmove` listener for this layer
    if (onLayerPosChanged) {
      konvaLayer.on('dragend', function (e) {
        onLayerPosChanged(rpLayer.id, e.target.x(), e.target.y());
      });
    }

    // The dragBoundFunc limits how far the layer can be dragged
    konvaLayer.dragBoundFunc(function (pos) {
      const cursorPos = getScaledCursorPosition(stage);
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

    // To achieve performant transparency, we use the `source-in` blending mode on a rect that covers the entire layer.
    // The brush strokes group functions as a mask for this rect, which has the layer's fill and opacity. The brush
    // strokes' color doesn't matter - the only requirement is that they are not transparent.
    const transparencyRect = new Konva.Rect({
      id: getRPLayerTransparencyRectId(rpLayer.id),
      globalCompositeOperation: 'source-in',
      listening: false,
    });
    konvaLayer.add(transparencyRect);

    stage.add(konvaLayer);

    // When a layer is added, it ends up on top of the brush preview - we need to move the preview back to the top.
    stage.findOne<Konva.Layer>(`#${BRUSH_PREVIEW_LAYER_ID}`)?.moveToTop();
  }

  // Update the layer's position and listening state (only the selected layer is listening)
  konvaLayer.setAttrs({
    listening: rpLayer.id === selectedLayerIdId && tool === 'move',
    x: rpLayer.x,
    y: rpLayer.y,
    // There are rpLayers.length layers, plus a brush preview layer rendered on top of them, so the zIndex works
    // out to be the layerIndex. If more layers are added, this may no longer be true.
    zIndex: rpLayerIndex,
  });

  const color = rgbColorToString(rpLayer.color);
  const konvaObjectGroup = konvaLayer.findOne<Konva.Group>(`.${REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME}`);
  assert(konvaObjectGroup, `Object group not found for layer ${rpLayer.id}`);
  const transparencyRect = konvaLayer.findOne<Konva.Rect>(`#${getRPLayerTransparencyRectId(rpLayer.id)}`);
  assert(transparencyRect, `Transparency rect not found for layer ${rpLayer.id}`);

  // Remove deleted objects
  const objectIds = rpLayer.objects.map(mapId);
  for (const objectNode of konvaLayer.find(`.${REGIONAL_PROMPT_LAYER_LINE_NAME}`)) {
    if (!objectIds.includes(objectNode.id())) {
      objectNode.destroy();
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
    }
    // Only update the color if it has changed.
    if (konvaObject.stroke() !== color) {
      konvaObject.stroke(color);
    }
    // Only update layer visibility if it has changed.
    if (konvaObject.visible() !== rpLayer.isVisible) {
      konvaObject.visible(rpLayer.isVisible);
    }
  }

  // Set the layer opacity - must happen after all objects are added to the layer so the rect is the right size
  transparencyRect.setAttrs({
    ...konvaLayer.getClientRect({ skipTransform: true }),
    fill: color,
    opacity: layerOpacity,
  });
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

const selectPromptLayerObjectGroup = (item: Node<NodeConfig>) =>
  item.name() !== REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME;

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
  tool: RPTool,
  selectedLayerIdId: string | null,
  onBboxChanged: (layerId: string, bbox: IRect) => void
) => {
  // Hide all bounding boxes
  for (const bboxRect of stage.find<Konva.Rect>(`.${REGIONAL_PROMPT_LAYER_BBOX_NAME}`)) {
    bboxRect.visible(false);
    bboxRect.listening(false);
  }

  // No selected layer or not using the move tool - nothing more to do here
  if (!selectedLayerIdId || tool !== 'move') {
    return;
  }

  const konvaLayer = stage.findOne<Konva.Layer>(`#${selectedLayerIdId}`);
  assert(konvaLayer, `Selected layer ${selectedLayerIdId} not found in stage`);

  const bbox = getKonvaLayerBbox(konvaLayer, selectPromptLayerObjectGroup);
  onBboxChanged(selectedLayerIdId, bbox);

  let rect = konvaLayer.findOne<Konva.Rect>(`.${REGIONAL_PROMPT_LAYER_BBOX_NAME}`);
  if (!rect) {
    rect = new Konva.Rect({
      id: getPRLayerBboxId(selectedLayerIdId),
      name: REGIONAL_PROMPT_LAYER_BBOX_NAME,
      strokeWidth: 1,
    });
    konvaLayer.add(rect);
  }
  rect.setAttrs({
    visible: true,
    x: bbox.x,
    y: bbox.y,
    width: bbox.width,
    height: bbox.height,
    listening: true,
    stroke: selectedLayerIdId === selectedLayerIdId ? 'rgba(153, 187, 189, 1)' : 'rgba(255, 255, 255, 0.149)',
  });
};
