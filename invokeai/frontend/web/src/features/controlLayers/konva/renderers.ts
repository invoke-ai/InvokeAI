import { rgbaColorToString, rgbColorToString } from 'features/canvas/util/colorToString';
import { getLayerBboxFast, getLayerBboxPixels } from 'features/controlLayers/konva/bbox';
import { LightnessToAlphaFilter } from 'features/controlLayers/konva/filters';
import {
  BACKGROUND_LAYER_ID,
  BACKGROUND_RECT_ID,
  CA_LAYER_IMAGE_NAME,
  CA_LAYER_NAME,
  COMPOSITING_RECT_NAME,
  getCALayerImageId,
  getIILayerImageId,
  getLayerBboxId,
  getRGLayerObjectGroupId,
  INITIAL_IMAGE_LAYER_IMAGE_NAME,
  INITIAL_IMAGE_LAYER_NAME,
  LAYER_BBOX_NAME,
  NO_LAYERS_MESSAGE_LAYER_ID,
  RG_LAYER_LINE_NAME,
  RG_LAYER_NAME,
  RG_LAYER_OBJECT_GROUP_NAME,
  RG_LAYER_RECT_NAME,
  TOOL_PREVIEW_BRUSH_BORDER_INNER_ID,
  TOOL_PREVIEW_BRUSH_BORDER_OUTER_ID,
  TOOL_PREVIEW_BRUSH_FILL_ID,
  TOOL_PREVIEW_BRUSH_GROUP_ID,
  TOOL_PREVIEW_LAYER_ID,
  TOOL_PREVIEW_RECT_ID,
} from 'features/controlLayers/konva/naming';
import { getScaledFlooredCursorPosition, snapPosToStage } from 'features/controlLayers/konva/util';
import {
  isControlAdapterLayer,
  isInitialImageLayer,
  isRegionalGuidanceLayer,
  isRenderableLayer,
} from 'features/controlLayers/store/controlLayersSlice';
import type {
  BrushLine,
  ControlAdapterLayer,
  EraserLine,
  InitialImageLayer,
  Layer,
  RectShape,
  RegionalGuidanceLayer,
  Tool,
} from 'features/controlLayers/store/types';
import { t } from 'i18next';
import Konva from 'konva';
import type { IRect, Vector2d } from 'konva/lib/types';
import { debounce } from 'lodash-es';
import type { RgbColor } from 'react-colorful';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import {
  BBOX_SELECTED_STROKE,
  BRUSH_BORDER_INNER_COLOR,
  BRUSH_BORDER_OUTER_COLOR,
  TRANSPARENCY_CHECKER_PATTERN,
} from './constants';

const mapId = (object: { id: string }): string => object.id;

/**
 * Konva selection callback to select all renderable layers. This includes RG, CA and II layers.
 */
const selectRenderableLayers = (n: Konva.Node): boolean =>
  n.name() === RG_LAYER_NAME || n.name() === CA_LAYER_NAME || n.name() === INITIAL_IMAGE_LAYER_NAME;

/**
 * Konva selection callback to select RG mask objects. This includes lines and rects.
 */
const selectVectorMaskObjects = (node: Konva.Node): boolean => {
  return node.name() === RG_LAYER_LINE_NAME || node.name() === RG_LAYER_RECT_NAME;
};

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
  const rectPreview = new Konva.Rect({ id: TOOL_PREVIEW_RECT_ID, listening: false, stroke: 'white', strokeWidth: 1 });
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
const renderToolPreview = (
  stage: Konva.Stage,
  tool: Tool,
  color: RgbColor | null,
  selectedLayerType: Layer['type'] | null,
  globalMaskLayerOpacity: number,
  cursorPos: Vector2d | null,
  lastMouseDownPos: Vector2d | null,
  brushSize: number
): void => {
  const layerCount = stage.find(selectRenderableLayers).length;
  // Update the stage's pointer style
  if (layerCount === 0) {
    // We have no layers, so we should not render any tool
    stage.container().style.cursor = 'default';
  } else if (selectedLayerType !== 'regional_guidance_layer') {
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
    const snappedPos = snapPosToStage(cursorPos, stage);
    const rectPreview = toolPreviewLayer.findOne<Konva.Rect>(`#${TOOL_PREVIEW_RECT_ID}`);
    rectPreview?.setAttrs({
      x: Math.min(snappedPos.x, lastMouseDownPos.x),
      y: Math.min(snappedPos.y, lastMouseDownPos.y),
      width: Math.abs(snappedPos.x - lastMouseDownPos.x),
      height: Math.abs(snappedPos.y - lastMouseDownPos.y),
    });
    rectPreview?.visible(true);
  } else {
    rectPreview?.visible(false);
  }
};

/**
 * Creates a regional guidance layer.
 * @param stage The konva stage
 * @param layerState The regional guidance layer state
 * @param onLayerPosChanged Callback for when the layer's position changes
 */
const createRGLayer = (
  stage: Konva.Stage,
  layerState: RegionalGuidanceLayer,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
): Konva.Layer => {
  // This layer hasn't been added to the konva state yet
  const konvaLayer = new Konva.Layer({
    id: layerState.id,
    name: RG_LAYER_NAME,
    draggable: true,
    dragDistance: 0,
  });

  // When a drag on the layer finishes, update the layer's position in state. During the drag, konva handles changing
  // the position - we do not need to call this on the `dragmove` event.
  if (onLayerPosChanged) {
    konvaLayer.on('dragend', function (e) {
      onLayerPosChanged(layerState.id, Math.floor(e.target.x()), Math.floor(e.target.y()));
    });
  }

  // The dragBoundFunc limits how far the layer can be dragged
  konvaLayer.dragBoundFunc(function (pos) {
    const cursorPos = getScaledFlooredCursorPosition(stage);
    if (!cursorPos) {
      return this.getAbsolutePosition();
    }
    // Prevent the user from dragging the layer out of the stage bounds by constaining the cursor position to the stage bounds
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
    id: getRGLayerObjectGroupId(layerState.id, uuidv4()),
    name: RG_LAYER_OBJECT_GROUP_NAME,
    listening: false,
  });
  konvaLayer.add(konvaObjectGroup);

  stage.add(konvaLayer);

  return konvaLayer;
};

/**
 * Creates a konva vector mask brush line from a vector mask line.
 * @param brushLine The vector mask line state
 * @param layerObjectGroup The konva layer's object group to add the line to
 */
const createVectorMaskBrushLine = (brushLine: BrushLine, layerObjectGroup: Konva.Group): Konva.Line => {
  const konvaLine = new Konva.Line({
    id: brushLine.id,
    key: brushLine.id,
    name: RG_LAYER_LINE_NAME,
    strokeWidth: brushLine.strokeWidth,
    tension: 0,
    lineCap: 'round',
    lineJoin: 'round',
    shadowForStrokeEnabled: false,
    globalCompositeOperation: 'source-over',
    listening: false,
  });
  layerObjectGroup.add(konvaLine);
  return konvaLine;
};

/**
 * Creates a konva vector mask eraser line from a vector mask line.
 * @param eraserLine The vector mask line state
 * @param layerObjectGroup The konva layer's object group to add the line to
 */
const createVectorMaskEraserLine = (eraserLine: EraserLine, layerObjectGroup: Konva.Group): Konva.Line => {
  const konvaLine = new Konva.Line({
    id: eraserLine.id,
    key: eraserLine.id,
    name: RG_LAYER_LINE_NAME,
    strokeWidth: eraserLine.strokeWidth,
    tension: 0,
    lineCap: 'round',
    lineJoin: 'round',
    shadowForStrokeEnabled: false,
    globalCompositeOperation: 'destination-out',
    listening: false,
  });
  layerObjectGroup.add(konvaLine);
  return konvaLine;
};

const createVectorMaskLine = (maskObject: BrushLine | EraserLine, layerObjectGroup: Konva.Group): Konva.Line => {
  if (maskObject.type === 'brush_line') {
    return createVectorMaskBrushLine(maskObject, layerObjectGroup);
  } else {
    // maskObject.type === 'eraser_line'
    return createVectorMaskEraserLine(maskObject, layerObjectGroup);
  }
};

/**
 * Creates a konva rect from a vector mask rect.
 * @param vectorMaskRect The vector mask rect state
 * @param layerObjectGroup The konva layer's object group to add the line to
 */
const createVectorMaskRect = (vectorMaskRect: RectShape, layerObjectGroup: Konva.Group): Konva.Rect => {
  const konvaRect = new Konva.Rect({
    id: vectorMaskRect.id,
    key: vectorMaskRect.id,
    name: RG_LAYER_RECT_NAME,
    x: vectorMaskRect.x,
    y: vectorMaskRect.y,
    width: vectorMaskRect.width,
    height: vectorMaskRect.height,
    listening: false,
  });
  layerObjectGroup.add(konvaRect);
  return konvaRect;
};

/**
 * Creates the "compositing rect" for a layer.
 * @param konvaLayer The konva layer
 */
const createCompositingRect = (konvaLayer: Konva.Layer): Konva.Rect => {
  const compositingRect = new Konva.Rect({ name: COMPOSITING_RECT_NAME, listening: false });
  konvaLayer.add(compositingRect);
  return compositingRect;
};

/**
 * Renders a regional guidance layer.
 * @param stage The konva stage
 * @param layerState The regional guidance layer state
 * @param globalMaskLayerOpacity The global mask layer opacity
 * @param tool The current tool
 * @param onLayerPosChanged Callback for when the layer's position changes
 */
const renderRGLayer = (
  stage: Konva.Stage,
  layerState: RegionalGuidanceLayer,
  globalMaskLayerOpacity: number,
  tool: Tool,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
): void => {
  const konvaLayer =
    stage.findOne<Konva.Layer>(`#${layerState.id}`) ?? createRGLayer(stage, layerState, onLayerPosChanged);

  // Update the layer's position and listening state
  konvaLayer.setAttrs({
    listening: tool === 'move', // The layer only listens when using the move tool - otherwise the stage is handling mouse events
    x: Math.floor(layerState.x),
    y: Math.floor(layerState.y),
  });

  // Convert the color to a string, stripping the alpha - the object group will handle opacity.
  const rgbColor = rgbColorToString(layerState.previewColor);

  const konvaObjectGroup = konvaLayer.findOne<Konva.Group>(`.${RG_LAYER_OBJECT_GROUP_NAME}`);
  assert(konvaObjectGroup, `Object group not found for layer ${layerState.id}`);

  // We use caching to handle "global" layer opacity, but caching is expensive and we should only do it when required.
  let groupNeedsCache = false;

  const objectIds = layerState.maskObjects.map(mapId);
  // Destroy any objects that are no longer in the redux state
  for (const objectNode of konvaObjectGroup.find(selectVectorMaskObjects)) {
    if (!objectIds.includes(objectNode.id())) {
      objectNode.destroy();
      groupNeedsCache = true;
    }
  }

  for (const maskObject of layerState.maskObjects) {
    if (maskObject.type === 'brush_line' || maskObject.type === 'eraser_line') {
      const vectorMaskLine =
        stage.findOne<Konva.Line>(`#${maskObject.id}`) ?? createVectorMaskLine(maskObject, konvaObjectGroup);

      // Only update the points if they have changed. The point values are never mutated, they are only added to the
      // array, so checking the length is sufficient to determine if we need to re-cache.
      if (vectorMaskLine.points().length !== maskObject.points.length) {
        vectorMaskLine.points(maskObject.points);
        groupNeedsCache = true;
      }
      // Only update the color if it has changed.
      if (vectorMaskLine.stroke() !== rgbColor) {
        vectorMaskLine.stroke(rgbColor);
        groupNeedsCache = true;
      }
    } else if (maskObject.type === 'rect_shape') {
      const konvaObject =
        stage.findOne<Konva.Rect>(`#${maskObject.id}`) ?? createVectorMaskRect(maskObject, konvaObjectGroup);

      // Only update the color if it has changed.
      if (konvaObject.fill() !== rgbColor) {
        konvaObject.fill(rgbColor);
        groupNeedsCache = true;
      }
    }
  }

  // Only update layer visibility if it has changed.
  if (konvaLayer.visible() !== layerState.isEnabled) {
    konvaLayer.visible(layerState.isEnabled);
    groupNeedsCache = true;
  }

  if (konvaObjectGroup.getChildren().length === 0) {
    // No objects - clear the cache to reset the previous pixel data
    konvaObjectGroup.clearCache();
    return;
  }

  const compositingRect =
    konvaLayer.findOne<Konva.Rect>(`.${COMPOSITING_RECT_NAME}`) ?? createCompositingRect(konvaLayer);

  /**
   * When the group is selected, we use a rect of the selected preview color, composited over the shapes. This allows
   * shapes to render as a "raster" layer with all pixels drawn at the same color and opacity.
   *
   * Without this special handling, each shape is drawn individually with the given opacity, atop the other shapes. The
   * effect is like if you have a Photoshop Group consisting of many shapes, each of which has the given opacity.
   * Overlapping shapes will have their colors blended together, and the final color is the result of all the shapes.
   *
   * Instead, with the special handling, the effect is as if you drew all the shapes at 100% opacity, flattened them to
   * a single raster image, and _then_ applied the 50% opacity.
   */
  if (layerState.isSelected && tool !== 'move') {
    // We must clear the cache first so Konva will re-draw the group with the new compositing rect
    if (konvaObjectGroup.isCached()) {
      konvaObjectGroup.clearCache();
    }
    // The user is allowed to reduce mask opacity to 0, but we need the opacity for the compositing rect to work
    konvaObjectGroup.opacity(1);

    compositingRect.setAttrs({
      // The rect should be the size of the layer - use the fast method if we don't have a pixel-perfect bbox already
      ...(!layerState.bboxNeedsUpdate && layerState.bbox ? layerState.bbox : getLayerBboxFast(konvaLayer)),
      fill: rgbColor,
      opacity: globalMaskLayerOpacity,
      // Draw this rect only where there are non-transparent pixels under it (e.g. the mask shapes)
      globalCompositeOperation: 'source-in',
      visible: true,
      // This rect must always be on top of all other shapes
      zIndex: konvaObjectGroup.getChildren().length,
    });
  } else {
    // The compositing rect should only be shown when the layer is selected.
    compositingRect.visible(false);
    // Cache only if needed - or if we are on this code path and _don't_ have a cache
    if (groupNeedsCache || !konvaObjectGroup.isCached()) {
      konvaObjectGroup.cache();
    }
    // Updating group opacity does not require re-caching
    konvaObjectGroup.opacity(globalMaskLayerOpacity);
  }
};

/**
 * Creates an initial image konva layer.
 * @param stage The konva stage
 * @param layerState The initial image layer state
 */
const createIILayer = (stage: Konva.Stage, layerState: InitialImageLayer): Konva.Layer => {
  const konvaLayer = new Konva.Layer({
    id: layerState.id,
    name: INITIAL_IMAGE_LAYER_NAME,
    imageSmoothingEnabled: true,
    listening: false,
  });
  stage.add(konvaLayer);
  return konvaLayer;
};

/**
 * Creates the konva image for an initial image layer.
 * @param konvaLayer The konva layer
 * @param imageEl The image element
 */
const createIILayerImage = (konvaLayer: Konva.Layer, imageEl: HTMLImageElement): Konva.Image => {
  const konvaImage = new Konva.Image({
    name: INITIAL_IMAGE_LAYER_IMAGE_NAME,
    image: imageEl,
  });
  konvaLayer.add(konvaImage);
  return konvaImage;
};

/**
 * Updates an initial image layer's attributes (width, height, opacity, visibility).
 * @param stage The konva stage
 * @param konvaImage The konva image
 * @param layerState The initial image layer state
 */
const updateIILayerImageAttrs = (stage: Konva.Stage, konvaImage: Konva.Image, layerState: InitialImageLayer): void => {
  // Konva erroneously reports NaN for width and height when the stage is hidden. This causes errors when caching,
  // but it doesn't seem to break anything.
  // TODO(psyche): Investigate and report upstream.
  const newWidth = stage.width() / stage.scaleX();
  const newHeight = stage.height() / stage.scaleY();
  if (
    konvaImage.width() !== newWidth ||
    konvaImage.height() !== newHeight ||
    konvaImage.visible() !== layerState.isEnabled
  ) {
    konvaImage.setAttrs({
      opacity: layerState.opacity,
      scaleX: 1,
      scaleY: 1,
      width: stage.width() / stage.scaleX(),
      height: stage.height() / stage.scaleY(),
      visible: layerState.isEnabled,
    });
  }
  if (konvaImage.opacity() !== layerState.opacity) {
    konvaImage.opacity(layerState.opacity);
  }
};

/**
 * Update an initial image layer's image source when the image changes.
 * @param stage The konva stage
 * @param konvaLayer The konva layer
 * @param layerState The initial image layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
const updateIILayerImageSource = async (
  stage: Konva.Stage,
  konvaLayer: Konva.Layer,
  layerState: InitialImageLayer,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): Promise<void> => {
  if (layerState.image) {
    const imageName = layerState.image.name;
    const imageDTO = await getImageDTO(imageName);
    if (!imageDTO) {
      return;
    }
    const imageEl = new Image();
    const imageId = getIILayerImageId(layerState.id, imageName);
    imageEl.onload = () => {
      // Find the existing image or create a new one - must find using the name, bc the id may have just changed
      const konvaImage =
        konvaLayer.findOne<Konva.Image>(`.${INITIAL_IMAGE_LAYER_IMAGE_NAME}`) ??
        createIILayerImage(konvaLayer, imageEl);

      // Update the image's attributes
      konvaImage.setAttrs({
        id: imageId,
        image: imageEl,
      });
      updateIILayerImageAttrs(stage, konvaImage, layerState);
      imageEl.id = imageId;
    };
    imageEl.src = imageDTO.image_url;
  } else {
    konvaLayer.findOne(`.${INITIAL_IMAGE_LAYER_IMAGE_NAME}`)?.destroy();
  }
};

/**
 * Renders an initial image layer.
 * @param stage The konva stage
 * @param layerState The initial image layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
const renderIILayer = (
  stage: Konva.Stage,
  layerState: InitialImageLayer,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): void => {
  const konvaLayer = stage.findOne<Konva.Layer>(`#${layerState.id}`) ?? createIILayer(stage, layerState);
  const konvaImage = konvaLayer.findOne<Konva.Image>(`.${INITIAL_IMAGE_LAYER_IMAGE_NAME}`);
  const canvasImageSource = konvaImage?.image();
  let imageSourceNeedsUpdate = false;
  if (canvasImageSource instanceof HTMLImageElement) {
    const image = layerState.image;
    if (image && canvasImageSource.id !== getCALayerImageId(layerState.id, image.name)) {
      imageSourceNeedsUpdate = true;
    } else if (!image) {
      imageSourceNeedsUpdate = true;
    }
  } else if (!canvasImageSource) {
    imageSourceNeedsUpdate = true;
  }

  if (imageSourceNeedsUpdate) {
    updateIILayerImageSource(stage, konvaLayer, layerState, getImageDTO);
  } else if (konvaImage) {
    updateIILayerImageAttrs(stage, konvaImage, layerState);
  }
};

/**
 * Creates a control adapter layer.
 * @param stage The konva stage
 * @param layerState The control adapter layer state
 */
const createCALayer = (stage: Konva.Stage, layerState: ControlAdapterLayer): Konva.Layer => {
  const konvaLayer = new Konva.Layer({
    id: layerState.id,
    name: CA_LAYER_NAME,
    imageSmoothingEnabled: true,
    listening: false,
  });
  stage.add(konvaLayer);
  return konvaLayer;
};

/**
 * Creates a control adapter layer image.
 * @param konvaLayer The konva layer
 * @param imageEl The image element
 */
const createCALayerImage = (konvaLayer: Konva.Layer, imageEl: HTMLImageElement): Konva.Image => {
  const konvaImage = new Konva.Image({
    name: CA_LAYER_IMAGE_NAME,
    image: imageEl,
  });
  konvaLayer.add(konvaImage);
  return konvaImage;
};

/**
 * Updates the image source for a control adapter layer. This includes loading the image from the server and updating the konva image.
 * @param stage The konva stage
 * @param konvaLayer The konva layer
 * @param layerState The control adapter layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
const updateCALayerImageSource = async (
  stage: Konva.Stage,
  konvaLayer: Konva.Layer,
  layerState: ControlAdapterLayer,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): Promise<void> => {
  const image = layerState.controlAdapter.processedImage ?? layerState.controlAdapter.image;
  if (image) {
    const imageName = image.name;
    const imageDTO = await getImageDTO(imageName);
    if (!imageDTO) {
      return;
    }
    const imageEl = new Image();
    const imageId = getCALayerImageId(layerState.id, imageName);
    imageEl.onload = () => {
      // Find the existing image or create a new one - must find using the name, bc the id may have just changed
      const konvaImage =
        konvaLayer.findOne<Konva.Image>(`.${CA_LAYER_IMAGE_NAME}`) ?? createCALayerImage(konvaLayer, imageEl);

      // Update the image's attributes
      konvaImage.setAttrs({
        id: imageId,
        image: imageEl,
      });
      updateCALayerImageAttrs(stage, konvaImage, layerState);
      // Must cache after this to apply the filters
      konvaImage.cache();
      imageEl.id = imageId;
    };
    imageEl.src = imageDTO.image_url;
  } else {
    konvaLayer.findOne(`.${CA_LAYER_IMAGE_NAME}`)?.destroy();
  }
};

/**
 * Updates the image attributes for a control adapter layer's image (width, height, visibility, opacity, filters).
 * @param stage The konva stage
 * @param konvaImage The konva image
 * @param layerState The control adapter layer state
 */
const updateCALayerImageAttrs = (
  stage: Konva.Stage,
  konvaImage: Konva.Image,
  layerState: ControlAdapterLayer
): void => {
  let needsCache = false;
  // Konva erroneously reports NaN for width and height when the stage is hidden. This causes errors when caching,
  // but it doesn't seem to break anything.
  // TODO(psyche): Investigate and report upstream.
  const newWidth = stage.width() / stage.scaleX();
  const newHeight = stage.height() / stage.scaleY();
  const hasFilter = konvaImage.filters() !== null && konvaImage.filters().length > 0;
  if (
    konvaImage.width() !== newWidth ||
    konvaImage.height() !== newHeight ||
    konvaImage.visible() !== layerState.isEnabled ||
    hasFilter !== layerState.isFilterEnabled
  ) {
    konvaImage.setAttrs({
      opacity: layerState.opacity,
      scaleX: 1,
      scaleY: 1,
      width: stage.width() / stage.scaleX(),
      height: stage.height() / stage.scaleY(),
      visible: layerState.isEnabled,
      filters: layerState.isFilterEnabled ? [LightnessToAlphaFilter] : [],
    });
    needsCache = true;
  }
  if (konvaImage.opacity() !== layerState.opacity) {
    konvaImage.opacity(layerState.opacity);
  }
  if (needsCache) {
    konvaImage.cache();
  }
};

/**
 * Renders a control adapter layer. If the layer doesn't already exist, it is created. Otherwise, the layer is updated
 * with the current image source and attributes.
 * @param stage The konva stage
 * @param layerState The control adapter layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
const renderCALayer = (
  stage: Konva.Stage,
  layerState: ControlAdapterLayer,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): void => {
  const konvaLayer = stage.findOne<Konva.Layer>(`#${layerState.id}`) ?? createCALayer(stage, layerState);
  const konvaImage = konvaLayer.findOne<Konva.Image>(`.${CA_LAYER_IMAGE_NAME}`);
  const canvasImageSource = konvaImage?.image();
  let imageSourceNeedsUpdate = false;
  if (canvasImageSource instanceof HTMLImageElement) {
    const image = layerState.controlAdapter.processedImage ?? layerState.controlAdapter.image;
    if (image && canvasImageSource.id !== getCALayerImageId(layerState.id, image.name)) {
      imageSourceNeedsUpdate = true;
    } else if (!image) {
      imageSourceNeedsUpdate = true;
    }
  } else if (!canvasImageSource) {
    imageSourceNeedsUpdate = true;
  }

  if (imageSourceNeedsUpdate) {
    updateCALayerImageSource(stage, konvaLayer, layerState, getImageDTO);
  } else if (konvaImage) {
    updateCALayerImageAttrs(stage, konvaImage, layerState);
  }
};

/**
 * Renders the layers on the stage.
 * @param stage The konva stage
 * @param layerStates Array of all layer states
 * @param globalMaskLayerOpacity The global mask layer opacity
 * @param tool The current tool
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 * @param onLayerPosChanged Callback for when the layer's position changes
 */
const renderLayers = (
  stage: Konva.Stage,
  layerStates: Layer[],
  globalMaskLayerOpacity: number,
  tool: Tool,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
): void => {
  const layerIds = layerStates.filter(isRenderableLayer).map(mapId);
  // Remove un-rendered layers
  for (const konvaLayer of stage.find<Konva.Layer>(selectRenderableLayers)) {
    if (!layerIds.includes(konvaLayer.id())) {
      konvaLayer.destroy();
    }
  }

  for (const layer of layerStates) {
    if (isRegionalGuidanceLayer(layer)) {
      renderRGLayer(stage, layer, globalMaskLayerOpacity, tool, onLayerPosChanged);
    }
    if (isControlAdapterLayer(layer)) {
      renderCALayer(stage, layer, getImageDTO);
    }
    if (isInitialImageLayer(layer)) {
      renderIILayer(stage, layer, getImageDTO);
    }
    // IP Adapter layers are not rendered
  }
};

/**
 * Creates a bounding box rect for a layer.
 * @param layerState The layer state for the layer to create the bounding box for
 * @param konvaLayer The konva layer to attach the bounding box to
 */
const createBboxRect = (layerState: Layer, konvaLayer: Konva.Layer): Konva.Rect => {
  const rect = new Konva.Rect({
    id: getLayerBboxId(layerState.id),
    name: LAYER_BBOX_NAME,
    strokeWidth: 1,
    visible: false,
  });
  konvaLayer.add(rect);
  return rect;
};

/**
 * Renders the bounding boxes for the layers.
 * @param stage The konva stage
 * @param layerStates An array of layers to draw bboxes for
 * @param tool The current tool
 * @returns
 */
const renderBboxes = (stage: Konva.Stage, layerStates: Layer[], tool: Tool): void => {
  // Hide all bboxes so they don't interfere with getClientRect
  for (const bboxRect of stage.find<Konva.Rect>(`.${LAYER_BBOX_NAME}`)) {
    bboxRect.visible(false);
    bboxRect.listening(false);
  }
  // No selected layer or not using the move tool - nothing more to do here
  if (tool !== 'move') {
    return;
  }

  for (const layer of layerStates.filter(isRegionalGuidanceLayer)) {
    if (!layer.bbox) {
      continue;
    }
    const konvaLayer = stage.findOne<Konva.Layer>(`#${layer.id}`);
    assert(konvaLayer, `Layer ${layer.id} not found in stage`);

    const bboxRect = konvaLayer.findOne<Konva.Rect>(`.${LAYER_BBOX_NAME}`) ?? createBboxRect(layer, konvaLayer);

    bboxRect.setAttrs({
      visible: !layer.bboxNeedsUpdate,
      listening: layer.isSelected,
      x: layer.bbox.x,
      y: layer.bbox.y,
      width: layer.bbox.width,
      height: layer.bbox.height,
      stroke: layer.isSelected ? BBOX_SELECTED_STROKE : '',
    });
  }
};

/**
 * Calculates the bbox of each regional guidance layer. Only calculates if the mask has changed.
 * @param stage The konva stage
 * @param layerStates An array of layers to calculate bboxes for
 * @param onBboxChanged Callback for when the bounding box changes
 */
const updateBboxes = (
  stage: Konva.Stage,
  layerStates: Layer[],
  onBboxChanged: (layerId: string, bbox: IRect | null) => void
): void => {
  for (const rgLayer of layerStates.filter(isRegionalGuidanceLayer)) {
    const konvaLayer = stage.findOne<Konva.Layer>(`#${rgLayer.id}`);
    assert(konvaLayer, `Layer ${rgLayer.id} not found in stage`);
    // We only need to recalculate the bbox if the layer has changed
    if (rgLayer.bboxNeedsUpdate) {
      const bboxRect = konvaLayer.findOne<Konva.Rect>(`.${LAYER_BBOX_NAME}`) ?? createBboxRect(rgLayer, konvaLayer);

      // Hide the bbox while we calculate the new bbox, else the bbox will be included in the calculation
      const visible = bboxRect.visible();
      bboxRect.visible(false);

      if (rgLayer.maskObjects.length === 0) {
        // No objects - no bbox to calculate
        onBboxChanged(rgLayer.id, null);
      } else {
        // Calculate the bbox by rendering the layer and checking its pixels
        onBboxChanged(rgLayer.id, getLayerBboxPixels(konvaLayer));
      }

      // Restore the visibility of the bbox
      bboxRect.visible(visible);
    }
  }
};

/**
 * Creates the background layer for the stage.
 * @param stage The konva stage
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
  image.src = TRANSPARENCY_CHECKER_PATTERN;
  return layer;
};

/**
 * Renders the background layer for the stage.
 * @param stage The konva stage
 * @param width The unscaled width of the canvas
 * @param height The unscaled height of the canvas
 */
const renderBackground = (stage: Konva.Stage, width: number, height: number): void => {
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

/**
 * Creates the "no layers" fallback layer
 * @param stage The konva stage
 */
const createNoLayersMessageLayer = (stage: Konva.Stage): Konva.Layer => {
  const noLayersMessageLayer = new Konva.Layer({
    id: NO_LAYERS_MESSAGE_LAYER_ID,
    opacity: 0.7,
    listening: false,
  });
  const text = new Konva.Text({
    x: 0,
    y: 0,
    align: 'center',
    verticalAlign: 'middle',
    text: t('controlLayers.noLayersAdded', 'No Layers Added'),
    fontFamily: '"Inter Variable", sans-serif',
    fontStyle: '600',
    fill: 'white',
  });
  noLayersMessageLayer.add(text);
  stage.add(noLayersMessageLayer);
  return noLayersMessageLayer;
};

/**
 * Renders the "no layers" message when there are no layers to render
 * @param stage The konva stage
 * @param layerCount The current number of layers
 * @param width The target width of the text
 * @param height The target height of the text
 */
const renderNoLayersMessage = (stage: Konva.Stage, layerCount: number, width: number, height: number): void => {
  const noLayersMessageLayer =
    stage.findOne<Konva.Layer>(`#${NO_LAYERS_MESSAGE_LAYER_ID}`) ?? createNoLayersMessageLayer(stage);
  if (layerCount === 0) {
    noLayersMessageLayer.findOne<Konva.Text>('Text')?.setAttrs({
      width,
      height,
      fontSize: 32 / stage.scaleX(),
    });
  } else {
    noLayersMessageLayer?.destroy();
  }
};

export const renderers = {
  renderToolPreview,
  renderLayers,
  renderBboxes,
  renderBackground,
  renderNoLayersMessage,
  arrangeLayers,
  updateBboxes,
};

const DEBOUNCE_MS = 300;

export const debouncedRenderers = {
  renderToolPreview: debounce(renderToolPreview, DEBOUNCE_MS),
  renderLayers: debounce(renderLayers, DEBOUNCE_MS),
  renderBboxes: debounce(renderBboxes, DEBOUNCE_MS),
  renderBackground: debounce(renderBackground, DEBOUNCE_MS),
  renderNoLayersMessage: debounce(renderNoLayersMessage, DEBOUNCE_MS),
  arrangeLayers: debounce(arrangeLayers, DEBOUNCE_MS),
  updateBboxes: debounce(updateBboxes, DEBOUNCE_MS),
};
