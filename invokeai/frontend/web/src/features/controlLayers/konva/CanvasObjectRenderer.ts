import type { JSONObject } from 'common/types';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import { CanvasBrushLineRenderer } from 'features/controlLayers/konva/CanvasBrushLine';
import { CanvasEraserLineRenderer } from 'features/controlLayers/konva/CanvasEraserLine';
import { CanvasImageRenderer } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
import { CanvasRectRenderer } from 'features/controlLayers/konva/CanvasRect';
import { getPatternSVG } from 'features/controlLayers/konva/patterns/getPatternSVG';
import { getPrefixedId, konvaNodeToBlob, konvaNodeToImageData, previewBlob } from 'features/controlLayers/konva/util';
import type {
  CanvasBrushLineState,
  CanvasEraserLineState,
  CanvasImageState,
  CanvasRectState,
  Fill,
  ImageCache,
  Rect,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { isEqual } from 'lodash-es';
import type { Logger } from 'roarr';
import { getImageDTO, uploadImage } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

function setFillPatternImage(shape: Konva.Shape, ...args: Parameters<typeof getPatternSVG>): HTMLImageElement {
  const imageElement = new Image();
  imageElement.onload = () => {
    shape.fillPatternImage(imageElement);
  };
  imageElement.src = getPatternSVG(...args);
  return imageElement;
}

/**
 * Union of all object renderers.
 */
type AnyObjectRenderer = CanvasBrushLineRenderer | CanvasEraserLineRenderer | CanvasRectRenderer | CanvasImageRenderer;
/**
 * Union of all object states.
 */
type AnyObjectState = CanvasBrushLineState | CanvasEraserLineState | CanvasImageState | CanvasRectState;

/**
 * Handles rendering of objects for a canvas entity.
 */
export class CanvasObjectRenderer {
  readonly type = 'object_renderer';

  id: string;
  path: string[];
  parent: CanvasLayerAdapter | CanvasMaskAdapter;
  manager: CanvasManager;
  log: Logger;

  /**
   * A set of subscriptions that should be cleaned up when the transformer is destroyed.
   */
  subscriptions: Set<() => void> = new Set();

  /**
   * A buffer object state that is rendered separately from the other objects. This is used for objects that are being
   * drawn in real-time, such as brush lines. The buffer object state only exists in this renderer and is not part of
   * the application state until it is committed.
   */
  buffer: AnyObjectState | null = null;

  /**
   * A map of object renderers, keyed by their ID.
   */
  renderers: Map<string, AnyObjectRenderer> = new Map();

  /**
   * A object containing singleton Konva nodes.
   */
  konva: {
    /**
     * A Konva Group that holds all the object renderers.
     */
    objectGroup: Konva.Group;
    /**
     * The compositing rect is used to draw the inpaint mask as a single shape with a given opacity.
     *
     * When drawing multiple transparent shapes on a canvas, overlapping regions will be more opaque. This doesn't
     * match the expectation for a mask, where all shapes should have the same opacity, even if they overlap.
     *
     * To prevent this, we use a trick. Instead of drawing all shapes at the desired opacity, we draw them at opacity of 1.
     * Then we draw a single rect that covers the entire canvas at the desired opacity, with a globalCompositeOperation
     * of 'source-in'. The shapes effectively become a mask for the "compositing rect".
     *
     * This node is only added when the parent of the renderer is an inpaint mask or region, which require this behavior.
     *
     * The compositing rect is not added to the object group.
     */
    compositing: {
      group: Konva.Group;
      rect: Konva.Rect;
      patternImage: HTMLImageElement;
    } | null;
  };

  constructor(parent: CanvasLayerAdapter | CanvasMaskAdapter) {
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.path = this.parent.path.concat(this.id);
    this.manager = parent.manager;
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.trace('Creating object renderer');

    this.konva = {
      objectGroup: new Konva.Group({ name: `${this.type}:object_group`, listening: false }),
      compositing: null,
    };

    this.parent.konva.layer.add(this.konva.objectGroup);

    if (this.parent.state.type === 'inpaint_mask' || this.parent.state.type === 'regional_guidance') {
      const rect = new Konva.Rect({
        name: `${this.type}:compositing_rect`,
        globalCompositeOperation: 'source-in',
        listening: false,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      });
      this.konva.compositing = {
        group: new Konva.Group({ name: `${this.type}:compositing_group`, listening: false }),
        rect,
        patternImage: new Image(), // we will set the src on this on the first render
      };
      this.konva.compositing.group.add(this.konva.compositing.rect);
      this.parent.konva.layer.add(this.konva.compositing.group);
    }

    this.subscriptions.add(
      this.manager.stateApi.$toolState.listen((newVal, oldVal) => {
        if (newVal.selected !== oldVal.selected) {
          this.commitBuffer();
        }
      })
    );

    // The compositing rect must cover the whole stage at all times. When the stage is scaled, moved or resized, we
    // need to update the compositing rect to match the stage.
    this.subscriptions.add(
      this.manager.stateApi.$stageAttrs.listen(() => {
        if (this.konva.compositing && this.parent.type === 'mask_adapter') {
          this.updateCompositingRectFill(this.parent.state.fill);
          this.updateCompositingRectSize();
        }
      })
    );
  }

  /**
   * Renders the given objects.
   * @param objectStates The objects to render.
   * @returns A promise that resolves to a boolean, indicating if any of the objects were rendered.
   */
  render = async (objectStates: AnyObjectState[]): Promise<boolean> => {
    let didRender = false;

    const objectIds = objectStates.map((objectState) => objectState.id);

    for (const renderer of this.renderers.values()) {
      if (!objectIds.includes(renderer.id) && renderer.id !== this.buffer?.id) {
        this.renderers.delete(renderer.id);
        renderer.destroy();
        didRender = true;
      }
    }

    for (const objectState of objectStates) {
      didRender = (await this.renderObject(objectState)) || didRender;
    }

    if (this.buffer) {
      didRender = (await this.renderObject(this.buffer)) || didRender;
    }

    this.syncCache(didRender);

    return didRender;
  };

  syncCache = (force: boolean = false) => {
    if (this.renderers.size === 0) {
      this.log.trace('Clearing object group cache');
      this.konva.objectGroup.clearCache();
    } else if (force || !this.konva.objectGroup.isCached()) {
      this.log.trace('Caching object group');
      this.konva.objectGroup.clearCache();
      this.konva.objectGroup.cache();
    }
  };

  updateCompositingRectFill = (fill: Fill) => {
    this.log.trace('Updating compositing rect fill');
    assert(this.konva.compositing, 'Missing compositing rect');

    if (fill.style === 'solid') {
      this.konva.compositing.rect.setAttrs({
        fill: rgbColorToString(fill.color),
        fillPriority: 'color',
      });
    } else {
      this.konva.compositing.rect.setAttrs({
        fillPriority: 'pattern',
      });
      setFillPatternImage(this.konva.compositing.rect, fill.style, fill.color);
    }
  };

  updateCompositingRectSize = () => {
    this.log.trace('Updating compositing rect size');
    assert(this.konva.compositing, 'Missing compositing rect');

    const { x, y, width, height, scale } = this.manager.stateApi.$stageAttrs.get();

    this.konva.compositing.rect.setAttrs({
      x: -x / scale,
      y: -y / scale,
      width: width / scale,
      height: height / scale,
      fillPatternScaleX: 1 / scale,
      fillPatternScaleY: 1 / scale,
    });
  };

  updateOpacity = (opacity: number) => {
    this.log.trace('Updating opacity');
    if (this.konva.compositing) {
      this.konva.compositing.group.opacity(opacity);
    } else {
      this.konva.objectGroup.opacity(opacity);
    }
  };

  /**
   * Renders the given object. If the object renderer does not exist, it will be created and its Konva group added to the
   * parent entity's object group.
   * @param objectState The object's state.
   * @param force Whether to force the object to render, even if it has not changed. If omitted, the object renderer
   * will only render if the object state has changed. The exception is the first render, where the object will always
   * be rendered.
   * @returns A promise that resolves to a boolean, indicating if the object was rendered.
   */
  renderObject = async (objectState: AnyObjectState, force = false): Promise<boolean> => {
    let didRender = false;

    let renderer = this.renderers.get(objectState.id);

    const isFirstRender = renderer === undefined;

    if (objectState.type === 'brush_line') {
      assert(renderer instanceof CanvasBrushLineRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasBrushLineRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'eraser_line') {
      assert(renderer instanceof CanvasEraserLineRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasEraserLineRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'rect') {
      assert(renderer instanceof CanvasRectRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasRectRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'image') {
      assert(renderer instanceof CanvasImageRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasImageRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.konva.objectGroup.add(renderer.konva.group);
      }
      didRender = await renderer.update(objectState, force || isFirstRender);
    }

    if (didRender && this.konva.objectGroup.isCached()) {
      this.konva.objectGroup.clearCache();
    }

    return didRender;
  };

  /**
   * Determines if the renderer has a buffer object to render.
   * @returns Whether the renderer has a buffer object to render.
   */
  hasBuffer = (): boolean => {
    return this.buffer !== null;
  };

  /**
   * Sets the buffer object state to render.
   * @param objectState The object state to set as the buffer.
   * @returns A promise that resolves to a boolean, indicating if the object was rendered.
   */
  setBuffer = async (objectState: AnyObjectState): Promise<boolean> => {
    this.log.trace('Setting buffer');

    this.buffer = objectState;
    return await this.renderObject(this.buffer, true);
  };

  /**
   * Clears the buffer object state.
   */
  clearBuffer = () => {
    this.log.trace('Clearing buffer');

    if (this.buffer) {
      const renderer = this.renderers.get(this.buffer.id);
      if (renderer) {
        this.log.trace('Destroying buffer object renderer');
        renderer.destroy();
        this.renderers.delete(renderer.id);
      }
    }

    this.buffer = null;
  };

  /**
   * Commits the current buffer object, pushing the buffer object state back to the application state.
   */
  commitBuffer = () => {
    if (!this.buffer) {
      this.log.trace('No buffer to commit');
      return;
    }

    this.log.trace('Committing buffer');

    // We need to give the objects a fresh ID else they will be considered the same object when they are re-rendered as
    // a non-buffer object, and we won't trigger things like bbox calculation
    this.buffer.id = getPrefixedId(this.buffer.type);

    if (this.buffer.type === 'brush_line') {
      this.manager.stateApi.addBrushLine({
        entityIdentifier: this.parent.getEntityIdentifier(),
        brushLine: this.buffer,
      });
    } else if (this.buffer.type === 'eraser_line') {
      this.manager.stateApi.addEraserLine({
        entityIdentifier: this.parent.getEntityIdentifier(),
        eraserLine: this.buffer,
      });
    } else if (this.buffer.type === 'rect') {
      this.manager.stateApi.addRect({ entityIdentifier: this.parent.getEntityIdentifier(), rect: this.buffer });
    } else {
      this.log.warn({ buffer: this.buffer }, 'Invalid buffer object type');
    }

    this.buffer = null;
  };

  hideObjects = (except: string[] = []) => {
    for (const renderer of this.renderers.values()) {
      renderer.setVisibility(except.includes(renderer.id));
    }
  };

  showObjects = (except: string[] = []) => {
    for (const renderer of this.renderers.values()) {
      renderer.setVisibility(!except.includes(renderer.id));
    }
  };

  /**
   * Determines if the objects in the renderer require a pixel bbox calculation.
   *
   * In some cases, we can use Konva's getClientRect as the bbox, but it is not always accurate. It includes
   * these visually transparent shapes in its calculation:
   *
   * - Eraser lines, which are normal lines with a globalCompositeOperation of 'destination-out'.
   * - Clipped portions of any shape.
   * - Images, which may have transparent areas.
   */
  needsPixelBbox = (): boolean => {
    let needsPixelBbox = false;
    for (const renderer of this.renderers.values()) {
      const isEraserLine = renderer instanceof CanvasEraserLineRenderer;
      const isImage = renderer instanceof CanvasImageRenderer;
      const hasClip = renderer instanceof CanvasBrushLineRenderer && renderer.state.clip;
      if (isEraserLine || hasClip || isImage) {
        needsPixelBbox = true;
        break;
      }
    }
    return needsPixelBbox;
  };

  /**
   * Checks if the renderer has any objects to render, including its buffer.
   * @returns Whether the renderer has any objects to render.
   */
  hasObjects = (): boolean => {
    return this.renderers.size > 0 || this.buffer !== null;
  };

  getRasterizedImageCache = (rect: Rect): ImageCache | null => {
    const imageCache = this.parent.state.rasterizationCache.find((cache) => isEqual(cache.rect, rect));
    return imageCache ?? null;
  };

  /**
   * Rasterizes the parent entity. If the entity has a rasterization cache for the given rect, the cached image is
   * returned. Otherwise, the entity is rasterized and the image is uploaded to the server.
   *
   * The rasterization cache is reset when the entity's state changes. The buffer object is not considered part of the
   * entity state for this purpose as it is a temporary object.
   *
   * @param rect The rect to rasterize. If omitted, the entity's full rect will be used.
   * @returns A promise that resolves to the rasterized image DTO.
   */
  rasterize = async (rect: Rect, replaceObjects: boolean = false): Promise<ImageDTO> => {
    let imageDTO: ImageDTO | null = null;
    const rasterizedImageCache = this.getRasterizedImageCache(rect);

    if (rasterizedImageCache) {
      imageDTO = await getImageDTO(rasterizedImageCache.imageName);
      if (imageDTO) {
        this.log.trace({ rect, rasterizedImageCache, imageDTO }, 'Using cached rasterized image');
        return imageDTO;
      }
    }

    this.log.trace({ rect }, 'Rasterizing entity');

    const blob = await this.getBlob(rect);
    if (this.manager._isDebugging) {
      previewBlob(blob, 'Rasterized entity');
    }
    imageDTO = await uploadImage(blob, `${this.id}_rasterized.png`, 'other', true);
    const imageObject = imageDTOToImageObject(imageDTO);
    if (replaceObjects) {
      await this.renderObject(imageObject, true);
    }
    this.manager.stateApi.rasterizeEntity({
      entityIdentifier: this.parent.getEntityIdentifier(),
      imageObject,
      rect: { x: Math.round(rect.x), y: Math.round(rect.y), width: imageDTO.width, height: imageDTO.height },
      replaceObjects,
    });

    return imageDTO;
  };

  getBlob = (rect?: Rect): Promise<Blob> => {
    return konvaNodeToBlob(this.konva.objectGroup.clone(), rect);
  };

  getImageData = (rect?: Rect): ImageData => {
    return konvaNodeToImageData(this.konva.objectGroup.clone(), rect);
  };

  /**
   * Destroys this renderer and all of its object renderers.
   */
  destroy = () => {
    this.log.trace('Destroying object renderer');
    for (const cleanup of this.subscriptions) {
      this.log.trace('Cleaning up listener');
      cleanup();
    }
    for (const renderer of this.renderers.values()) {
      renderer.destroy();
    }
    this.renderers.clear();
  };

  /**
   * Gets a serializable representation of the renderer.
   * @returns A serializable representation of the renderer.
   */
  repr = () => {
    return {
      id: this.id,
      type: this.type,
      parent: this.parent.id,
      renderers: Array.from(this.renderers.values()).map((renderer) => renderer.repr()),
      buffer: deepClone(this.buffer),
    };
  };

  getLoggingContext = (): JSONObject => {
    return { ...this.parent.getLoggingContext(), path: this.path.join('.') };
  };
}
