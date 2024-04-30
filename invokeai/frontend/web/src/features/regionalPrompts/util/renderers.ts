import { getStore } from 'app/store/nanostores/store';
import { rgbaColorToString, rgbColorToString } from 'features/canvas/util/colorToString';
import { getScaledFlooredCursorPosition } from 'features/regionalPrompts/hooks/mouseEventHooks';
import {
  $tool,
  BACKGROUND_LAYER_ID,
  BACKGROUND_RECT_ID,
  CONTROLNET_LAYER_IMAGE_NAME,
  CONTROLNET_LAYER_NAME,
  getControlNetLayerImageId,
  getLayerBboxId,
  getMaskedGuidanceLayerObjectGroupId,
  isControlAdapterLayer,
  isMaskedGuidanceLayer,
  isRenderableLayer,
  LAYER_BBOX_NAME,
  MASKED_GUIDANCE_LAYER_LINE_NAME,
  MASKED_GUIDANCE_LAYER_NAME,
  MASKED_GUIDANCE_LAYER_OBJECT_GROUP_NAME,
  MASKED_GUIDANCE_LAYER_RECT_NAME,
  TOOL_PREVIEW_BRUSH_BORDER_INNER_ID,
  TOOL_PREVIEW_BRUSH_BORDER_OUTER_ID,
  TOOL_PREVIEW_BRUSH_FILL_ID,
  TOOL_PREVIEW_BRUSH_GROUP_ID,
  TOOL_PREVIEW_LAYER_ID,
  TOOL_PREVIEW_RECT_ID,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import type {
  ControlAdapterLayer,
  Layer,
  MaskedGuidanceLayer,
  Tool,
  VectorMaskLine,
  VectorMaskRect,
} from 'features/regionalPrompts/store/types';
import { getLayerBboxFast, getLayerBboxPixels } from 'features/regionalPrompts/util/bbox';
import Konva from 'konva';
import type { IRect, Vector2d } from 'konva/lib/types';
import { debounce } from 'lodash-es';
import type { RgbColor } from 'react-colorful';
import { imagesApi } from 'services/api/endpoints/images';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

const BBOX_SELECTED_STROKE = 'rgba(78, 190, 255, 1)';
const BRUSH_BORDER_INNER_COLOR = 'rgba(0,0,0,1)';
const BRUSH_BORDER_OUTER_COLOR = 'rgba(255,255,255,0.8)';
// This is invokeai/frontend/web/public/assets/images/transparent_bg.png as a dataURL
const STAGE_BG_DATAURL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAAEsmlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4KPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNS41LjAiPgogPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgeG1sbnM6ZXhpZj0iaHR0cDovL25zLmFkb2JlLmNvbS9leGlmLzEuMC8iCiAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyIKICAgIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIKICAgIHhtbG5zOnhtcD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLyIKICAgIHhtbG5zOnhtcE1NPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvbW0vIgogICAgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIKICAgZXhpZjpQaXhlbFhEaW1lbnNpb249IjIwIgogICBleGlmOlBpeGVsWURpbWVuc2lvbj0iMjAiCiAgIGV4aWY6Q29sb3JTcGFjZT0iMSIKICAgdGlmZjpJbWFnZVdpZHRoPSIyMCIKICAgdGlmZjpJbWFnZUxlbmd0aD0iMjAiCiAgIHRpZmY6UmVzb2x1dGlvblVuaXQ9IjIiCiAgIHRpZmY6WFJlc29sdXRpb249IjMwMC8xIgogICB0aWZmOllSZXNvbHV0aW9uPSIzMDAvMSIKICAgcGhvdG9zaG9wOkNvbG9yTW9kZT0iMyIKICAgcGhvdG9zaG9wOklDQ1Byb2ZpbGU9InNSR0IgSUVDNjE5NjYtMi4xIgogICB4bXA6TW9kaWZ5RGF0ZT0iMjAyNC0wNC0yM1QwODoyMDo0NysxMDowMCIKICAgeG1wOk1ldGFkYXRhRGF0ZT0iMjAyNC0wNC0yM1QwODoyMDo0NysxMDowMCI+CiAgIDx4bXBNTTpIaXN0b3J5PgogICAgPHJkZjpTZXE+CiAgICAgPHJkZjpsaQogICAgICBzdEV2dDphY3Rpb249InByb2R1Y2VkIgogICAgICBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZmZpbml0eSBQaG90byAxLjEwLjgiCiAgICAgIHN0RXZ0OndoZW49IjIwMjQtMDQtMjNUMDg6MjA6NDcrMTA6MDAiLz4KICAgIDwvcmRmOlNlcT4KICAgPC94bXBNTTpIaXN0b3J5PgogIDwvcmRmOkRlc2NyaXB0aW9uPgogPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KPD94cGFja2V0IGVuZD0iciI/Pn9pdVgAAAGBaUNDUHNSR0IgSUVDNjE5NjYtMi4xAAAokXWR3yuDURjHP5uJmKghFy6WxpVpqMWNMgm1tGbKr5vt3S+1d3t73y3JrXKrKHHj1wV/AbfKtVJESq53TdywXs9rakv2nJ7zfM73nOfpnOeAPZJRVMPhAzWb18NTAffC4pK7oYiDTjpw4YgqhjYeCgWpaR8P2Kx457Vq1T73rzXHE4YCtkbhMUXT88LTwsG1vGbxrnC7ko7Ghc+F+3W5oPC9pcfKXLQ4VeYvi/VIeALsbcLuVBXHqlhJ66qwvByPmikov/exXuJMZOfnJPaId2MQZooAbmaYZAI/g4zK7MfLEAOyoka+7yd/lpzkKjJrrKOzSoo0efpFLUj1hMSk6AkZGdat/v/tq5EcHipXdwag/sU033qhYQdK26b5eWyapROoe4arbCU/dwQj76JvVzTPIbRuwsV1RYvtweUWdD1pUT36I9WJ25NJeD2DlkVw3ULTcrlnv/ucPkJkQ77qBvYPoE/Ot658AxagZ8FoS/a7AAAACXBIWXMAAC4jAAAuIwF4pT92AAAAL0lEQVQ4jWM8ffo0A25gYmKCR5YJjxxBMKp5ZGhm/P//Px7pM2fO0MrmUc0jQzMAB2EIhZC3pUYAAAAASUVORK5CYII=';

const mapId = (object: { id: string }) => object.id;

const selectRenderableLayers = (n: Konva.Node) =>
  n.name() === MASKED_GUIDANCE_LAYER_NAME || n.name() === CONTROLNET_LAYER_NAME;

const selectVectorMaskObjects = (node: Konva.Node) => {
  return node.name() === MASKED_GUIDANCE_LAYER_LINE_NAME || node.name() === MASKED_GUIDANCE_LAYER_RECT_NAME;
};

/**
 * Creates the brush preview layer.
 * @param stage The konva stage to render on.
 * @returns The brush preview layer.
 */
const createToolPreviewLayer = (stage: Konva.Stage) => {
  // Initialize the brush preview layer & add to the stage
  const toolPreviewLayer = new Konva.Layer({ id: TOOL_PREVIEW_LAYER_ID, visible: false, listening: false });
  stage.add(toolPreviewLayer);

  // Add handlers to show/hide the brush preview layer
  stage.on('mousemove', (e) => {
    const tool = $tool.get();
    e.target
      .getStage()
      ?.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)
      ?.visible(tool === 'brush' || tool === 'eraser');
  });
  stage.on('mouseleave', (e) => {
    e.target.getStage()?.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.visible(false);
  });
  stage.on('mouseenter', (e) => {
    const tool = $tool.get();
    e.target
      .getStage()
      ?.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)
      ?.visible(tool === 'brush' || tool === 'eraser');
  });

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

  // Create the rect preview
  const rectPreview = new Konva.Rect({ id: TOOL_PREVIEW_RECT_ID, listening: false, stroke: 'white', strokeWidth: 1 });
  toolPreviewLayer.add(rectPreview);

  return toolPreviewLayer;
};

/**
 * Renders the brush preview for the selected tool.
 * @param stage The konva stage to render on.
 * @param tool The selected tool.
 * @param color The selected layer's color.
 * @param cursorPos The cursor position.
 * @param lastMouseDownPos The position of the last mouse down event - used for the rect tool.
 * @param brushSize The brush size.
 */
const renderToolPreview = (
  stage: Konva.Stage,
  tool: Tool,
  color: RgbColor | null,
  globalMaskLayerOpacity: number,
  cursorPos: Vector2d | null,
  lastMouseDownPos: Vector2d | null,
  isMouseOver: boolean,
  brushSize: number
) => {
  const layerCount = stage.find(`.${MASKED_GUIDANCE_LAYER_NAME}`).length;
  // Update the stage's pointer style
  if (layerCount === 0) {
    // We have no layers, so we should not render any tool
    stage.container().style.cursor = 'default';
  } else if (tool === 'move') {
    // Move tool gets a pointer
    stage.container().style.cursor = 'default';
  } else if (tool === 'rect') {
    // Move rect gets a crosshair
    stage.container().style.cursor = 'crosshair';
  } else {
    // Else we use the brush preview
    stage.container().style.cursor = 'none';
  }

  const toolPreviewLayer = stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`) ?? createToolPreviewLayer(stage);

  if (!isMouseOver || layerCount === 0) {
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
  if (cursorPos && color && (tool === 'brush' || tool === 'eraser')) {
    // Update the fill circle
    const brushPreviewFill = brushPreviewGroup.findOne<Konva.Circle>(`#${TOOL_PREVIEW_BRUSH_FILL_ID}`);
    brushPreviewFill?.setAttrs({
      x: cursorPos.x,
      y: cursorPos.y,
      radius: brushSize / 2,
      fill: rgbaColorToString({ ...color, a: globalMaskLayerOpacity }),
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
    const rectPreview = toolPreviewLayer.findOne<Konva.Rect>(`#${TOOL_PREVIEW_RECT_ID}`);
    rectPreview?.setAttrs({
      x: Math.min(cursorPos.x, lastMouseDownPos.x),
      y: Math.min(cursorPos.y, lastMouseDownPos.y),
      width: Math.abs(cursorPos.x - lastMouseDownPos.x),
      height: Math.abs(cursorPos.y - lastMouseDownPos.y),
    });
    rectPreview?.visible(true);
  } else {
    rectPreview?.visible(false);
  }
};

/**
 * Creates a vector mask layer.
 * @param stage The konva stage to attach the layer to.
 * @param reduxLayer The redux layer to create the konva layer from.
 * @param onLayerPosChanged Callback for when the layer's position changes.
 */
const createMaskedGuidanceLayer = (
  stage: Konva.Stage,
  reduxLayer: MaskedGuidanceLayer,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
) => {
  // This layer hasn't been added to the konva state yet
  const konvaLayer = new Konva.Layer({
    id: reduxLayer.id,
    name: MASKED_GUIDANCE_LAYER_NAME,
    draggable: true,
    dragDistance: 0,
  });

  // Create a `dragmove` listener for this layer
  if (onLayerPosChanged) {
    konvaLayer.on('dragend', function (e) {
      onLayerPosChanged(reduxLayer.id, Math.floor(e.target.x()), Math.floor(e.target.y()));
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
    id: getMaskedGuidanceLayerObjectGroupId(reduxLayer.id, uuidv4()),
    name: MASKED_GUIDANCE_LAYER_OBJECT_GROUP_NAME,
    listening: false,
  });
  konvaLayer.add(konvaObjectGroup);

  stage.add(konvaLayer);

  return konvaLayer;
};

/**
 * Creates a konva line from a redux vector mask line.
 * @param reduxObject The redux object to create the konva line from.
 * @param konvaGroup The konva group to add the line to.
 */
const createVectorMaskLine = (reduxObject: VectorMaskLine, konvaGroup: Konva.Group): Konva.Line => {
  const vectorMaskLine = new Konva.Line({
    id: reduxObject.id,
    key: reduxObject.id,
    name: MASKED_GUIDANCE_LAYER_LINE_NAME,
    strokeWidth: reduxObject.strokeWidth,
    tension: 0,
    lineCap: 'round',
    lineJoin: 'round',
    shadowForStrokeEnabled: false,
    globalCompositeOperation: reduxObject.tool === 'brush' ? 'source-over' : 'destination-out',
    listening: false,
  });
  konvaGroup.add(vectorMaskLine);
  return vectorMaskLine;
};

/**
 * Creates a konva rect from a redux vector mask rect.
 * @param reduxObject The redux object to create the konva rect from.
 * @param konvaGroup The konva group to add the rect to.
 */
const createVectorMaskRect = (reduxObject: VectorMaskRect, konvaGroup: Konva.Group): Konva.Rect => {
  const vectorMaskRect = new Konva.Rect({
    id: reduxObject.id,
    key: reduxObject.id,
    name: MASKED_GUIDANCE_LAYER_RECT_NAME,
    x: reduxObject.x,
    y: reduxObject.y,
    width: reduxObject.width,
    height: reduxObject.height,
    listening: false,
  });
  konvaGroup.add(vectorMaskRect);
  return vectorMaskRect;
};

/**
 * Renders a vector mask layer.
 * @param stage The konva stage to render on.
 * @param reduxLayer The redux vector mask layer to render.
 * @param reduxLayerIndex The index of the layer in the redux store.
 * @param globalMaskLayerOpacity The opacity of the global mask layer.
 * @param tool The current tool.
 */
const renderMaskedGuidanceLayer = (
  stage: Konva.Stage,
  reduxLayer: MaskedGuidanceLayer,
  globalMaskLayerOpacity: number,
  tool: Tool,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
): void => {
  const konvaLayer =
    stage.findOne<Konva.Layer>(`#${reduxLayer.id}`) ?? createMaskedGuidanceLayer(stage, reduxLayer, onLayerPosChanged);

  // Update the layer's position and listening state
  konvaLayer.setAttrs({
    listening: tool === 'move', // The layer only listens when using the move tool - otherwise the stage is handling mouse events
    x: Math.floor(reduxLayer.x),
    y: Math.floor(reduxLayer.y),
  });

  // Convert the color to a string, stripping the alpha - the object group will handle opacity.
  const rgbColor = rgbColorToString(reduxLayer.previewColor);

  const konvaObjectGroup = konvaLayer.findOne<Konva.Group>(`.${MASKED_GUIDANCE_LAYER_OBJECT_GROUP_NAME}`);
  assert(konvaObjectGroup, `Object group not found for layer ${reduxLayer.id}`);

  // We use caching to handle "global" layer opacity, but caching is expensive and we should only do it when required.
  let groupNeedsCache = false;

  const objectIds = reduxLayer.maskObjects.map(mapId);
  for (const objectNode of konvaObjectGroup.find(selectVectorMaskObjects)) {
    if (!objectIds.includes(objectNode.id())) {
      objectNode.destroy();
      groupNeedsCache = true;
    }
  }

  for (const reduxObject of reduxLayer.maskObjects) {
    if (reduxObject.type === 'vector_mask_line') {
      const vectorMaskLine =
        stage.findOne<Konva.Line>(`#${reduxObject.id}`) ?? createVectorMaskLine(reduxObject, konvaObjectGroup);

      // Only update the points if they have changed. The point values are never mutated, they are only added to the
      // array, so checking the length is sufficient to determine if we need to re-cache.
      if (vectorMaskLine.points().length !== reduxObject.points.length) {
        vectorMaskLine.points(reduxObject.points);
        groupNeedsCache = true;
      }
      // Only update the color if it has changed.
      if (vectorMaskLine.stroke() !== rgbColor) {
        vectorMaskLine.stroke(rgbColor);
        groupNeedsCache = true;
      }
    } else if (reduxObject.type === 'vector_mask_rect') {
      const konvaObject =
        stage.findOne<Konva.Rect>(`#${reduxObject.id}`) ?? createVectorMaskRect(reduxObject, konvaObjectGroup);

      // Only update the color if it has changed.
      if (konvaObject.fill() !== rgbColor) {
        konvaObject.fill(rgbColor);
        groupNeedsCache = true;
      }
    }
  }

  // Only update layer visibility if it has changed.
  if (konvaLayer.visible() !== reduxLayer.isEnabled) {
    konvaLayer.visible(reduxLayer.isEnabled);
    groupNeedsCache = true;
  }

  if (konvaObjectGroup.children.length === 0) {
    // No objects - clear the cache to reset the previous pixel data
    konvaObjectGroup.clearCache();
  } else if (groupNeedsCache) {
    konvaObjectGroup.cache();
  }

  // Updating group opacity does not require re-caching
  if (konvaObjectGroup.opacity() !== globalMaskLayerOpacity) {
    konvaObjectGroup.opacity(globalMaskLayerOpacity);
  }
};

const createControlNetLayer = (stage: Konva.Stage, reduxLayer: ControlAdapterLayer): Konva.Layer => {
  const konvaLayer = new Konva.Layer({
    id: reduxLayer.id,
    name: CONTROLNET_LAYER_NAME,
    imageSmoothingEnabled: true,
  });
  stage.add(konvaLayer);
  return konvaLayer;
};

const createControlNetLayerImage = (konvaLayer: Konva.Layer, image: HTMLImageElement): Konva.Image => {
  const konvaImage = new Konva.Image({
    name: CONTROLNET_LAYER_IMAGE_NAME,
    image,
    filters: [LightnessToAlphaFilter],
  });
  konvaLayer.add(konvaImage);
  return konvaImage;
};

const updateControlNetLayerImageSource = async (
  stage: Konva.Stage,
  konvaLayer: Konva.Layer,
  reduxLayer: ControlAdapterLayer
) => {
  if (reduxLayer.imageName) {
    const imageName = reduxLayer.imageName;
    const req = getStore().dispatch(imagesApi.endpoints.getImageDTO.initiate(reduxLayer.imageName));
    const imageDTO = await req.unwrap();
    req.unsubscribe();
    const image = new Image();
    const imageId = getControlNetLayerImageId(reduxLayer.id, imageName);
    image.onload = () => {
      // Find the existing image or create a new one - must find using the name, bc the id may have just changed
      const konvaImage =
        konvaLayer.findOne<Konva.Image>(`.${CONTROLNET_LAYER_IMAGE_NAME}`) ??
        createControlNetLayerImage(konvaLayer, image);

      // Update the image's attributes
      konvaImage.setAttrs({
        id: imageId,
        image,
      });
      updateControlNetLayerImageAttrs(stage, konvaImage, reduxLayer);
      // Must cache after this to apply the filters
      konvaImage.cache();
      image.id = imageId;
    };
    image.src = imageDTO.image_url;
  } else {
    konvaLayer.findOne(`.${CONTROLNET_LAYER_IMAGE_NAME}`)?.destroy();
  }
};

const updateControlNetLayerImageAttrs = (
  stage: Konva.Stage,
  konvaImage: Konva.Image,
  reduxLayer: ControlAdapterLayer
) => {
  let needsCache = false;
  const newWidth = stage.width() / stage.scaleX();
  const newHeight = stage.height() / stage.scaleY();
  if (
    konvaImage.width() !== newWidth ||
    konvaImage.height() !== newHeight ||
    konvaImage.visible() !== reduxLayer.isEnabled
  ) {
    konvaImage.setAttrs({
      opacity: reduxLayer.opacity,
      scaleX: 1,
      scaleY: 1,
      width: stage.width() / stage.scaleX(),
      height: stage.height() / stage.scaleY(),
      visible: reduxLayer.isEnabled,
    });
    needsCache = true;
  }
  if (konvaImage.opacity() !== reduxLayer.opacity) {
    konvaImage.opacity(reduxLayer.opacity);
  }
  if (needsCache) {
    konvaImage.cache();
  }
};

const renderControlNetLayer = (stage: Konva.Stage, reduxLayer: ControlAdapterLayer) => {
  const konvaLayer = stage.findOne<Konva.Layer>(`#${reduxLayer.id}`) ?? createControlNetLayer(stage, reduxLayer);
  const konvaImage = konvaLayer.findOne<Konva.Image>(`.${CONTROLNET_LAYER_IMAGE_NAME}`);
  const canvasImageSource = konvaImage?.image();
  let imageSourceNeedsUpdate = false;
  if (canvasImageSource instanceof HTMLImageElement) {
    if (
      reduxLayer.imageName &&
      canvasImageSource.id !== getControlNetLayerImageId(reduxLayer.id, reduxLayer.imageName)
    ) {
      imageSourceNeedsUpdate = true;
    } else if (!reduxLayer.imageName) {
      imageSourceNeedsUpdate = true;
    }
  } else if (!canvasImageSource) {
    imageSourceNeedsUpdate = true;
  }

  if (imageSourceNeedsUpdate) {
    updateControlNetLayerImageSource(stage, konvaLayer, reduxLayer);
  } else if (konvaImage) {
    updateControlNetLayerImageAttrs(stage, konvaImage, reduxLayer);
  }
};

/**
 * Renders the layers on the stage.
 * @param stage The konva stage to render on.
 * @param reduxLayers Array of the layers from the redux store.
 * @param layerOpacity The opacity of the layer.
 * @param onLayerPosChanged Callback for when the layer's position changes. This is optional to allow for offscreen rendering.
 * @returns
 */
const renderLayers = (
  stage: Konva.Stage,
  reduxLayers: Layer[],
  globalMaskLayerOpacity: number,
  tool: Tool,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
) => {
  const reduxLayerIds = reduxLayers.filter(isRenderableLayer).map(mapId);
  // Remove un-rendered layers
  for (const konvaLayer of stage.find<Konva.Layer>(selectRenderableLayers)) {
    if (!reduxLayerIds.includes(konvaLayer.id())) {
      konvaLayer.destroy();
    }
  }

  for (const reduxLayer of reduxLayers) {
    if (isMaskedGuidanceLayer(reduxLayer)) {
      renderMaskedGuidanceLayer(stage, reduxLayer, globalMaskLayerOpacity, tool, onLayerPosChanged);
    }
    if (isControlAdapterLayer(reduxLayer)) {
      renderControlNetLayer(stage, reduxLayer);
    }
  }
};

/**
 * Creates a bounding box rect for a layer.
 * @param reduxLayer The redux layer to create the bounding box for.
 * @param konvaLayer The konva layer to attach the bounding box to.
 * @param onBboxMouseDown Callback for when the bounding box is clicked.
 */
const createBboxRect = (reduxLayer: Layer, konvaLayer: Konva.Layer) => {
  const rect = new Konva.Rect({
    id: getLayerBboxId(reduxLayer.id),
    name: LAYER_BBOX_NAME,
    strokeWidth: 1,
  });
  konvaLayer.add(rect);
  return rect;
};

/**
 * Renders the bounding boxes for the layers.
 * @param stage The konva stage to render on
 * @param reduxLayers An array of all redux layers to draw bboxes for
 * @param selectedLayerId The selected layer's id
 * @param tool The current tool
 * @param onBboxChanged Callback for when the bbox is changed
 * @param onBboxMouseDown Callback for when the bbox is clicked
 * @returns
 */
const renderBbox = (
  stage: Konva.Stage,
  reduxLayers: Layer[],
  tool: Tool,
  onBboxChanged: (layerId: string, bbox: IRect | null) => void
) => {
  // Hide all bboxes so they don't interfere with getClientRect
  for (const bboxRect of stage.find<Konva.Rect>(`.${LAYER_BBOX_NAME}`)) {
    bboxRect.visible(false);
    bboxRect.listening(false);
  }
  // No selected layer or not using the move tool - nothing more to do here
  if (tool !== 'move') {
    return;
  }

  for (const reduxLayer of reduxLayers) {
    if (reduxLayer.type === 'masked_guidance_layer') {
      const konvaLayer = stage.findOne<Konva.Layer>(`#${reduxLayer.id}`);
      assert(konvaLayer, `Layer ${reduxLayer.id} not found in stage`);

      let bbox = reduxLayer.bbox;

      // We only need to recalculate the bbox if the layer has changed and it has objects
      if (reduxLayer.bboxNeedsUpdate && reduxLayer.maskObjects.length) {
        // We only need to use the pixel-perfect bounding box if the layer has eraser strokes
        bbox = reduxLayer.needsPixelBbox ? getLayerBboxPixels(konvaLayer) : getLayerBboxFast(konvaLayer);
        // Update the layer's bbox in the redux store
        onBboxChanged(reduxLayer.id, bbox);
      }

      if (!bbox) {
        continue;
      }

      const rect = konvaLayer.findOne<Konva.Rect>(`.${LAYER_BBOX_NAME}`) ?? createBboxRect(reduxLayer, konvaLayer);

      rect.setAttrs({
        visible: true,
        listening: reduxLayer.isSelected,
        x: bbox.x,
        y: bbox.y,
        width: bbox.width,
        height: bbox.height,
        stroke: reduxLayer.isSelected ? BBOX_SELECTED_STROKE : '',
      });
    }
  }
};

/**
 * Creates the background layer for the stage.
 * @param stage The konva stage to render on
 */
const createBackgroundLayer = (stage: Konva.Stage): Konva.Layer => {
  const layer = new Konva.Layer({
    id: BACKGROUND_LAYER_ID,
  });
  const background = new Konva.Rect({
    id: BACKGROUND_RECT_ID,
    x: stage.x(),
    y: 0,
    width: stage.width() / stage.scaleX(),
    height: stage.height() / stage.scaleY(),
    listening: false,
    opacity: 0.2,
  });
  layer.add(background);
  stage.add(layer);
  const image = new Image();
  image.onload = () => {
    background.fillPatternImage(image);
  };
  image.src = STAGE_BG_DATAURL;
  return layer;
};

/**
 * Renders the background layer for the stage.
 * @param stage The konva stage to render on
 * @param width The unscaled width of the canvas
 * @param height The unscaled height of the canvas
 */
const renderBackground = (stage: Konva.Stage, width: number, height: number) => {
  const layer = stage.findOne<Konva.Layer>(`#${BACKGROUND_LAYER_ID}`) ?? createBackgroundLayer(stage);

  const background = layer.findOne<Konva.Rect>(`#${BACKGROUND_RECT_ID}`);
  assert(background, 'Background rect not found');
  // ensure background rect is in the top-left of the canvas
  background.absolutePosition({ x: 0, y: 0 });

  // set the dimensions of the background rect to match the canvas - not the stage!!!
  background.size({
    width: width / stage.scaleX(),
    height: height / stage.scaleY(),
  });

  // Calculate the amount the stage is moved - including the effect of scaling
  const stagePos = {
    x: -stage.x() / stage.scaleX(),
    y: -stage.y() / stage.scaleY(),
  };

  // Apply that movement to the fill pattern
  background.fillPatternOffset(stagePos);
};

/**
 * Arranges all layers in the z-axis by updating their z-indices.
 * @param stage The konva stage
 * @param layerIds An array of redux layer ids, in their z-index order
 */
const arrangeLayers = (stage: Konva.Stage, layerIds: string[]): void => {
  let nextZIndex = 0;
  // Background is the first layer
  stage.findOne<Konva.Layer>(`#${BACKGROUND_LAYER_ID}`)?.zIndex(nextZIndex++);
  // Then arrange the redux layers in order
  for (const layerId of layerIds) {
    stage.findOne<Konva.Layer>(`#${layerId}`)?.zIndex(nextZIndex++);
  }
  // Finally, the tool preview layer is always on top
  stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.zIndex(nextZIndex++);
};

export const renderers = {
  renderToolPreview,
  renderLayers,
  renderBbox,
  renderBackground,
  arrangeLayers,
};

const DEBOUNCE_MS = 300;

export const debouncedRenderers = {
  renderToolPreview: debounce(renderToolPreview, DEBOUNCE_MS),
  renderLayers: debounce(renderLayers, DEBOUNCE_MS),
  renderBbox: debounce(renderBbox, DEBOUNCE_MS),
  renderBackground: debounce(renderBackground, DEBOUNCE_MS),
  arrangeLayers: debounce(arrangeLayers, DEBOUNCE_MS),
};

/**
 * Calculates the lightness (HSL) of a given pixel and sets the alpha channel to that value.
 * This is useful for edge maps and other masks, to make the black areas transparent.
 * @param imageData The image data to apply the filter to
 */
const LightnessToAlphaFilter = (imageData: ImageData) => {
  const len = imageData.data.length / 4;
  for (let i = 0; i < len; i++) {
    const r = imageData.data[i * 4 + 0] as number;
    const g = imageData.data[i * 4 + 1] as number;
    const b = imageData.data[i * 4 + 2] as number;
    const cMin = Math.min(r, g, b);
    const cMax = Math.max(r, g, b);
    imageData.data[i * 4 + 3] = (cMin + cMax) / 2;
  }
};
