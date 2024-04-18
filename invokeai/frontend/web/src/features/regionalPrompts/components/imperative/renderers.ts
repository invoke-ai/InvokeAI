import { rgbColorToString } from 'features/canvas/util/colorToString';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import type { Layer, Tool } from 'features/regionalPrompts/store/regionalPromptsSlice';
import {
  BRUSH_PREVIEW_BORDER_INNER_ID,
  BRUSH_PREVIEW_BORDER_OUTER_ID,
  BRUSH_PREVIEW_FILL_ID,
  BRUSH_PREVIEW_LAYER_ID,
  getPromptRegionLayerBboxId,
  getPromptRegionLayerObjectGroupId,
  getPromptRegionLayerTransparencyRectId,
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
  tool: Tool,
  color: RgbColor,
  cursorPos: Vector2d,
  brushSize: number
) => {
  // Update the stage's pointer style
  stage.container().style.cursor = tool === 'move' ? 'default' : 'none';

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

  // The brush preview is hidden when using the move tool
  layer.visible(tool !== 'move');

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

/**
 * Renders the layers on the stage.
 * @param stage The konva stage to render on.
 * @param reduxLayers Array of the layers from the redux store.
 * @param selectedLayerId The selected layer id.
 * @param layerOpacity The opacity of the layer.
 * @param onLayerPosChanged Callback for when the layer's position changes. This is optional to allow for offscreen rendering.
 * @returns
 */
export const renderLayers = (
  stage: Konva.Stage,
  reduxLayers: Layer[],
  selectedLayerId: string | null,
  layerOpacity: number,
  tool: Tool,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
) => {
  const reduxLayerIds = reduxLayers.map((l) => l.id);

  // Remove un-rendered layers
  for (const konvaLayer of stage.find<Konva.Layer>(`.${REGIONAL_PROMPT_LAYER_NAME}`)) {
    if (!reduxLayerIds.includes(konvaLayer.id())) {
      konvaLayer.destroy();
      console.log(`Destroyed layer ${konvaLayer.id()}`);
    }
  }

  for (const reduxLayer of reduxLayers) {
    let konvaLayer = stage.findOne<Konva.Layer>(`#${reduxLayer.id}`);

    if (!konvaLayer) {
      // This layer hasn't been added to the konva state yet
      konvaLayer = new Konva.Layer({
        id: reduxLayer.id,
        name: REGIONAL_PROMPT_LAYER_NAME,
        draggable: true,
      });

      // Create a `dragmove` listener for this layer
      if (onLayerPosChanged) {
        konvaLayer.on('dragend', function (e) {
          onLayerPosChanged(reduxLayer.id, e.target.x(), e.target.y());
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
        id: getPromptRegionLayerObjectGroupId(reduxLayer.id, uuidv4()),
        name: REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME,
        listening: false,
      });
      konvaLayer.add(konvaObjectGroup);

      // To achieve performant transparency, we use the `source-in` blending mode on a rect that covers the entire layer.
      // The brush strokes group functions as a mask for this rect, which has the layer's fill and opacity. The brush
      // strokes' color doesn't matter - the only requirement is that they are not transparent.
      const transparencyRect = new Konva.Rect({
        id: getPromptRegionLayerTransparencyRectId(reduxLayer.id),
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
      listening: reduxLayer.id === selectedLayerId && tool === 'move',
      x: reduxLayer.x,
      y: reduxLayer.y,
    });

    const color = rgbColorToString(reduxLayer.color);
    const konvaObjectGroup = konvaLayer.findOne<Konva.Group>(`.${REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME}`);
    assert(konvaObjectGroup, `Object group not found for layer ${reduxLayer.id}`);
    const transparencyRect = konvaLayer.findOne<Konva.Rect>(
      `#${getPromptRegionLayerTransparencyRectId(reduxLayer.id)}`
    );
    assert(transparencyRect, `Transparency rect not found for layer ${reduxLayer.id}`);

    // Remove deleted objects
    const objectIds = reduxLayer.objects.map((o) => o.id);
    for (const objectNode of konvaLayer.find(`.${REGIONAL_PROMPT_LAYER_LINE_NAME}`)) {
      if (!objectIds.includes(objectNode.id())) {
        objectNode.destroy();
      }
    }

    for (const reduxObject of reduxLayer.objects) {
      // TODO: Handle rects, images, etc
      if (reduxObject.kind !== 'line') {
        return;
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
      if (konvaObject.visible() !== reduxLayer.isVisible) {
        konvaObject.visible(reduxLayer.isVisible);
      }
    }

    // Set the layer opacity - must happen after all objects are added to the layer so the rect is the right size
    transparencyRect.setAttrs({
      ...konvaLayer.getClientRect({ skipTransform: true }),
      fill: color,
      opacity: layerOpacity,
    });
  }
};

const selectPromptLayerObjectGroup = (item: Node<NodeConfig>) =>
  item.name() !== REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME;

/**
 *
 * @param stage The konva stage to render on.
 * @param tool The current tool.
 * @param selectedLayerId The currently selected layer id.
 * @param onBboxChanged A callback to be called when the bounding box changes.
 * @returns
 */
export const renderBbox = (
  stage: Konva.Stage,
  tool: Tool,
  selectedLayerId: string | null,
  onBboxChanged: (layerId: string, bbox: IRect) => void
) => {
  // Hide all bounding boxes
  for (const bboxRect of stage.find<Konva.Rect>(`.${REGIONAL_PROMPT_LAYER_BBOX_NAME}`)) {
    bboxRect.visible(false);
    bboxRect.listening(false);
  }

  // No selected layer or not using the move tool - nothing more to do here
  if (!selectedLayerId || tool !== 'move') {
    return;
  }

  const konvaLayer = stage.findOne<Konva.Layer>(`#${selectedLayerId}`);
  assert(konvaLayer, `Selected layer ${selectedLayerId} not found in stage`);

  const bbox = getKonvaLayerBbox(konvaLayer, selectPromptLayerObjectGroup);
  onBboxChanged(selectedLayerId, bbox);

  let rect = konvaLayer.findOne<Konva.Rect>(`.${REGIONAL_PROMPT_LAYER_BBOX_NAME}`);
  if (!rect) {
    rect = new Konva.Rect({
      id: getPromptRegionLayerBboxId(selectedLayerId),
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
    stroke: selectedLayerId === selectedLayerId ? 'rgba(153, 187, 189, 1)' : 'rgba(255, 255, 255, 0.149)',
  });
};
