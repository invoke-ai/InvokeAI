import { rgbColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasEntityLayerAdapter } from 'features/controlLayers/konva/CanvasEntityLayerAdapter';
import type { CanvasEntityMaskAdapter } from 'features/controlLayers/konva/CanvasEntityMaskAdapter';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleABC } from 'features/controlLayers/konva/CanvasModuleABC';
import { CanvasObjectBrushLineRenderer } from 'features/controlLayers/konva/CanvasObjectBrushLineRenderer';
import { CanvasObjectEraserLineRenderer } from 'features/controlLayers/konva/CanvasObjectEraserLineRenderer';
import { CanvasObjectImageRenderer } from 'features/controlLayers/konva/CanvasObjectImageRenderer';
import { CanvasObjectRectRenderer } from 'features/controlLayers/konva/CanvasObjectRectRenderer';
import { LightnessToAlphaFilter } from 'features/controlLayers/konva/filters';
import { getPatternSVG } from 'features/controlLayers/konva/patterns/getPatternSVG';
import {
  getPrefixedId,
  konvaNodeToBlob,
  konvaNodeToCanvas,
  konvaNodeToImageData,
  previewBlob,
} from 'features/controlLayers/konva/util';
import type {
  CanvasBrushLineState,
  CanvasEraserLineState,
  CanvasImageState,
  CanvasRectState,
  Fill,
  Rect,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { GroupConfig } from 'konva/lib/Group';
import { debounce } from 'lodash-es';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { serializeError } from 'serialize-error';
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
type AnyObjectRenderer =
  | CanvasObjectBrushLineRenderer
  | CanvasObjectEraserLineRenderer
  | CanvasObjectRectRenderer
  | CanvasObjectImageRenderer;
/**
 * Union of all object states.
 */
type AnyObjectState = CanvasBrushLineState | CanvasEraserLineState | CanvasImageState | CanvasRectState;

/**
 * Handles rendering of objects for a canvas entity.
 */
export class CanvasEntityRenderer extends CanvasModuleABC {
  readonly type = 'entity_renderer';

  id: string;
  path: string[];
  parent: CanvasEntityLayerAdapter | CanvasEntityMaskAdapter;
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
  bufferState: AnyObjectState | null = null;

  /**
   * The object renderer for the buffer object state. It is created when the buffer object state is set and destroyed
   * when the buffer object state is cleared. This is separate from the other object renderers to allow the buffer to
   * be rendered separately.
   */
  bufferRenderer: AnyObjectRenderer | null = null;
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
     * A Konva Group that holds the buffer object renderer.
     */
    bufferGroup: Konva.Group;
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

  $canvasCache = atom<{ canvas: HTMLCanvasElement; rect: Rect } | null>(null);

  constructor(parent: CanvasEntityLayerAdapter | CanvasEntityMaskAdapter) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.path = this.parent.path.concat(this.id);
    this.manager = parent.manager;
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug('Creating entity object renderer module');

    this.konva = {
      objectGroup: new Konva.Group({ name: `${this.type}:object_group`, listening: false }),
      bufferGroup: new Konva.Group({ name: `${this.type}:buffer_group`, listening: false }),
      compositing: null,
    };

    this.parent.konva.layer.add(this.konva.objectGroup);
    this.parent.konva.layer.add(this.konva.bufferGroup);

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

    // When switching tool, commit the buffer. This is necessary to prevent the buffer from being lost when the
    // user switches tool mid-drawing, for example by pressing space to pan the stage. It's easy to press space
    // to pan _before_ releasing the mouse button, which would cause the buffer to be lost if we didn't commit it.
    this.subscriptions.add(
      this.manager.stateApi.$tool.listen(() => {
        this.commitBuffer();
      })
    );

    // The compositing rect must cover the whole stage at all times. When the stage is scaled, moved or resized, we
    // need to update the compositing rect to match the stage.
    this.subscriptions.add(
      this.manager.stateApi.$stageAttrs.listen(() => {
        if (this.konva.compositing && this.parent.type === 'entity_mask_adapter') {
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
      if (!objectIds.includes(renderer.id)) {
        this.renderers.delete(renderer.id);
        renderer.destroy();
        didRender = true;
      }
    }

    for (const objectState of objectStates) {
      didRender = (await this.renderObject(objectState)) || didRender;
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
      this.konva.objectGroup.cache({ pixelRatio: 1 });
      this.parent.renderer.updatePreviewCanvas();
    }
  };

  updateTransparencyEffect = (withTransparencyEffect: boolean) => {
    const filters = withTransparencyEffect ? [LightnessToAlphaFilter] : [];
    this.konva.objectGroup.filters(filters);
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
      this.konva.bufferGroup.opacity(opacity);
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

    const isFirstRender = !renderer;

    if (objectState.type === 'brush_line') {
      assert(renderer instanceof CanvasObjectBrushLineRenderer || !renderer);

      if (!renderer) {
        renderer = new CanvasObjectBrushLineRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'eraser_line') {
      assert(renderer instanceof CanvasObjectEraserLineRenderer || !renderer);

      if (!renderer) {
        renderer = new CanvasObjectEraserLineRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'rect') {
      assert(renderer instanceof CanvasObjectRectRenderer || !renderer);

      if (!renderer) {
        renderer = new CanvasObjectRectRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'image') {
      assert(renderer instanceof CanvasObjectImageRenderer || !renderer);

      if (!renderer) {
        renderer = new CanvasObjectImageRenderer(objectState, this);
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
   * Renders the buffer object. If the buffer renderer does not exist, it will be created and its Konva group added to the
   * parent entity's buffer object group.
   * @returns A promise that resolves to a boolean, indicating if the object was rendered.
   */
  renderBufferObject = async (): Promise<boolean> => {
    let didRender = false;

    if (!this.bufferState) {
      return false;
    }

    if (this.bufferState.type === 'brush_line') {
      assert(this.bufferRenderer instanceof CanvasObjectBrushLineRenderer || !this.bufferRenderer);

      if (!this.bufferRenderer) {
        this.bufferRenderer = new CanvasObjectBrushLineRenderer(this.bufferState, this);
        this.konva.bufferGroup.add(this.bufferRenderer.konva.group);
      }

      didRender = this.bufferRenderer.update(this.bufferState, true);
    } else if (this.bufferState.type === 'eraser_line') {
      assert(this.bufferRenderer instanceof CanvasObjectEraserLineRenderer || !this.bufferRenderer);

      if (!this.bufferRenderer) {
        this.bufferRenderer = new CanvasObjectEraserLineRenderer(this.bufferState, this);
        this.konva.bufferGroup.add(this.bufferRenderer.konva.group);
      }

      didRender = this.bufferRenderer.update(this.bufferState, true);
    } else if (this.bufferState.type === 'rect') {
      assert(this.bufferRenderer instanceof CanvasObjectRectRenderer || !this.bufferRenderer);

      if (!this.bufferRenderer) {
        this.bufferRenderer = new CanvasObjectRectRenderer(this.bufferState, this);
        this.konva.bufferGroup.add(this.bufferRenderer.konva.group);
      }

      didRender = this.bufferRenderer.update(this.bufferState, true);
    } else if (this.bufferState.type === 'image') {
      assert(this.bufferRenderer instanceof CanvasObjectImageRenderer || !this.bufferRenderer);

      if (!this.bufferRenderer) {
        this.bufferRenderer = new CanvasObjectImageRenderer(this.bufferState, this);
        this.konva.bufferGroup.add(this.bufferRenderer.konva.group);
      }
      didRender = await this.bufferRenderer.update(this.bufferState, true);
    }

    return didRender;
  };

  /**
   * Determines if the renderer has a buffer object to render.
   * @returns Whether the renderer has a buffer object to render.
   */
  hasBuffer = (): boolean => {
    return this.bufferState !== null || this.bufferRenderer !== null;
  };

  /**
   * Sets the buffer object state to render.
   * @param objectState The object state to set as the buffer.
   * @param resetBufferOffset Whether to reset the buffer's offset to 0,0. This is necessary when previewing filters.
   * When previewing a filter, the buffer object is an image of the same size as the entity, so it should be rendered
   * at the top-left corner of the entity.
   * @returns A promise that resolves to a boolean, indicating if the object was rendered.
   */
  setBuffer = async (objectState: AnyObjectState, resetBufferOffset: boolean = false): Promise<boolean> => {
    this.log.trace('Setting buffer');

    this.bufferState = objectState;
    if (resetBufferOffset) {
      this.konva.bufferGroup.offset({ x: 0, y: 0 });
    }
    return await this.renderBufferObject();
  };

  /**
   * Clears the buffer object state.
   */
  clearBuffer = () => {
    if (this.bufferState || this.bufferRenderer) {
      this.log.trace('Clearing buffer');
      this.bufferRenderer?.destroy();
      this.bufferRenderer = null;
      this.bufferState = null;
    }
  };

  /**
   * Commits the current buffer object, pushing the buffer object state back to the application state.
   */
  commitBuffer = (options?: { pushToState?: boolean }) => {
    const { pushToState } = { ...options, pushToState: true };

    if (!this.bufferState || !this.bufferRenderer) {
      this.log.trace('No buffer to commit');
      return;
    }

    this.log.trace('Committing buffer');

    // Move the buffer to the persistent objects group/renderers
    this.bufferRenderer.konva.group.moveTo(this.konva.objectGroup);
    this.renderers.set(this.bufferState.id, this.bufferRenderer);

    if (pushToState) {
      const entityIdentifier = this.parent.getEntityIdentifier();
      if (this.bufferState.type === 'brush_line') {
        this.manager.stateApi.addBrushLine({ entityIdentifier, brushLine: this.bufferState });
      } else if (this.bufferState.type === 'eraser_line') {
        this.manager.stateApi.addEraserLine({ entityIdentifier, eraserLine: this.bufferState });
      } else if (this.bufferState.type === 'rect') {
        this.manager.stateApi.addRect({ entityIdentifier, rect: this.bufferState });
      } else {
        this.log.warn({ buffer: this.bufferState }, 'Invalid buffer object type');
      }
    }

    this.bufferRenderer = null;
    this.bufferState = null;
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
      const isEraserLine = renderer instanceof CanvasObjectEraserLineRenderer;
      const isImage = renderer instanceof CanvasObjectImageRenderer;
      const hasClip = renderer instanceof CanvasObjectBrushLineRenderer && renderer.state.clip;
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
    return this.renderers.size > 0 || this.bufferState !== null || this.bufferRenderer !== null;
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
  rasterize = async (options: { rect: Rect; replaceObjects?: boolean; attrs?: GroupConfig }): Promise<ImageDTO> => {
    const { rect, replaceObjects, attrs } = { replaceObjects: false, attrs: {}, ...options };
    let imageDTO: ImageDTO | null = null;
    const hash = this.parent.hash({ rect, attrs });
    const cachedImageName = this.manager.cache.imageNameCache.get(hash);

    if (cachedImageName) {
      imageDTO = await getImageDTO(cachedImageName);
      if (imageDTO) {
        this.log.trace({ rect, cachedImageName, imageDTO }, 'Using cached rasterized image');
        return imageDTO;
      }
    }

    this.log.trace({ rect }, 'Rasterizing entity');

    const blob = await this.getBlob(rect, attrs);
    if (this.manager._isDebugging) {
      previewBlob(blob, 'Rasterized entity');
    }
    imageDTO = await uploadImage(blob, `${this.id}_rasterized.png`, 'other', true);
    const imageObject = imageDTOToImageObject(imageDTO);
    if (replaceObjects) {
      await this.setBuffer(imageObject);
      this.commitBuffer({ pushToState: false });
    }
    this.manager.stateApi.rasterizeEntity({
      entityIdentifier: this.parent.getEntityIdentifier(),
      imageObject,
      rect: { x: Math.round(rect.x), y: Math.round(rect.y), width: imageDTO.width, height: imageDTO.height },
      replaceObjects,
    });
    this.manager.cache.imageNameCache.set(hash, imageDTO.image_name);

    return imageDTO;
  };

  updatePreviewCanvas = debounce(() => {
    if (this.parent.transformer.isPendingRectCalculation) {
      return;
    }
    if (this.parent.transformer.pixelRect.width === 0 || this.parent.transformer.pixelRect.height === 0) {
      return;
    }
    try {
      const canvas = this.konva.objectGroup._getCachedSceneCanvas()._canvas as HTMLCanvasElement | undefined | null;
      if (canvas) {
        const nodeRect = this.parent.transformer.nodeRect;
        const pixelRect = this.parent.transformer.pixelRect;
        const rect = {
          x: pixelRect.x - nodeRect.x,
          y: pixelRect.y - nodeRect.y,
          width: pixelRect.width,
          height: pixelRect.height,
        };
        this.$canvasCache.set({ rect, canvas });
      }
    } catch (error) {
      // We are using an internal Konva method, so we need to catch any errors that may occur.
      this.log.warn({ error: serializeError(error) }, 'Failed to update preview canvas');
    }
  }, 300);

  cloneObjectGroup = (attrs?: GroupConfig): Konva.Group => {
    const clone = this.konva.objectGroup.clone();
    clone.cache();
    if (attrs) {
      clone.setAttrs(attrs);
    }
    return clone;
  };

  getCanvas = (rect?: Rect, attrs?: GroupConfig): HTMLCanvasElement => {
    const clone = this.cloneObjectGroup(attrs);
    const canvas = konvaNodeToCanvas(clone, rect);
    clone.destroy();
    return canvas;
  };

  getBlob = async (rect?: Rect, attrs?: GroupConfig): Promise<Blob> => {
    const clone = this.cloneObjectGroup(attrs);
    const blob = await konvaNodeToBlob(clone, rect);
    clone.destroy();
    return blob;
  };

  getImageData = (rect?: Rect, attrs?: GroupConfig): ImageData => {
    const clone = this.cloneObjectGroup(attrs);
    const imageData = konvaNodeToImageData(clone, rect);
    clone.destroy();
    return imageData;
  };

  /**
   * Destroys this renderer and all of its object renderers.
   */
  destroy = () => {
    this.log.debug('Destroying entity object renderer module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
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
      path: this.path,
      parent: this.parent.id,
      renderers: Array.from(this.renderers.values()).map((renderer) => renderer.repr()),
      buffer: this.bufferRenderer?.repr(),
    };
  };

  getLoggingContext = () => {
    return { ...this.parent.getLoggingContext(), path: this.path.join('.') };
  };
}
