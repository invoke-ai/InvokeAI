import { rgbColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntityAdapter/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectBrushLine } from 'features/controlLayers/konva/CanvasObjectBrushLine';
import { CanvasObjectEraserLine } from 'features/controlLayers/konva/CanvasObjectEraserLine';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObjectImage';
import { CanvasObjectRect } from 'features/controlLayers/konva/CanvasObjectRect';
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
type AnyObjectRenderer = CanvasObjectBrushLine | CanvasObjectEraserLine | CanvasObjectRect | CanvasObjectImage;
/**
 * Union of all object states.
 */
type AnyObjectState = CanvasBrushLineState | CanvasEraserLineState | CanvasImageState | CanvasRectState;

/**
 * Handles rendering of objects for a canvas entity.
 */
export class CanvasEntityObjectRenderer extends CanvasModuleBase {
  readonly type = 'object_renderer';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityAdapter;
  readonly manager: CanvasManager;
  readonly log: Logger;

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

  /**
   * The entity's object group as a canvas element along with the pixel rect of the entity at the time the canvas was
   * drawn.
   *
   * Technically, this is an internal Konva object, created when a Konva node's `.cache()` method is called. We cache
   * the object group after every update, so we get this as a "free" side effect.
   *
   * This is used to render the entity's preview in the control layer.
   */
  $canvasCache = atom<{ canvas: HTMLCanvasElement; rect: Rect } | null>(null);

  constructor(parent: CanvasEntityAdapter) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);
    this.log.debug('Creating module');

    this.konva = {
      objectGroup: new Konva.Group({ name: `${this.type}:object_group`, listening: false }),
      bufferGroup: new Konva.Group({ name: `${this.type}:buffer_group`, listening: false }),
      compositing: null,
    };

    this.parent.konva.layer.add(this.konva.objectGroup);
    this.parent.konva.layer.add(this.konva.bufferGroup);

    if (
      this.parent.entityIdentifier.type === 'inpaint_mask' ||
      this.parent.entityIdentifier.type === 'regional_guidance'
    ) {
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
      this.manager.tool.$tool.listen(() => {
        if (this.hasBuffer() && !this.manager.$isBusy.get()) {
          this.commitBuffer();
        }
      })
    );

    // The compositing rect must cover the whole stage at all times. When the stage is scaled, moved or resized, we
    // need to update the compositing rect to match the stage.
    this.subscriptions.add(
      this.manager.stage.$stageAttrs.listen(() => {
        if (
          this.konva.compositing &&
          (this.parent.type === 'inpaint_mask_adapter' || this.parent.type === 'regional_guidance_adapter')
        ) {
          this.updateCompositingRectSize();
        }
      })
    );
  }

  initialize = async () => {
    this.log.debug('Initializing module');
    await this.render();
  };

  /**
   * Renders the entity's objects.
   * @returns A promise that resolves to a boolean, indicating if any of the objects were rendered.
   */
  render = async (): Promise<boolean> => {
    let didRender = false;

    const objects = this.parent.state.objects;
    const objectIds = objects.map((obj) => obj.id);

    for (const renderer of this.renderers.values()) {
      if (!objectIds.includes(renderer.id)) {
        this.renderers.delete(renderer.id);
        renderer.destroy();
        didRender = true;
      }
    }

    for (const obj of objects) {
      didRender = (await this.renderObject(obj)) || didRender;
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

  updateTransparencyEffect = () => {
    if (this.parent.state.type === 'control_layer') {
      const filters = this.parent.state.withTransparencyEffect ? [LightnessToAlphaFilter] : [];
      this.konva.objectGroup.filters(filters);
    }
  };

  updateCompositingRectFill = () => {
    this.log.trace('Updating compositing rect fill');

    assert(this.konva.compositing, 'Missing compositing rect');
    assert(this.parent.state.type === 'inpaint_mask' || this.parent.state.type === 'regional_guidance');

    const fill = this.parent.state.fill;

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

    const { x, y, width, height, scale } = this.manager.stage.$stageAttrs.get();

    this.konva.compositing.rect.setAttrs({
      x: -x / scale,
      y: -y / scale,
      width: width / scale,
      height: height / scale,
      fillPatternScaleX: 1 / scale,
      fillPatternScaleY: 1 / scale,
    });
  };

  updateOpacity = () => {
    this.log.trace('Updating opacity');

    const opacity = this.manager.stateApi.getIsTypeHidden(this.parent.entityIdentifier.type)
      ? 0
      : this.parent.state.opacity;

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
      assert(renderer instanceof CanvasObjectBrushLine || !renderer);

      if (!renderer) {
        renderer = new CanvasObjectBrushLine(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'eraser_line') {
      assert(renderer instanceof CanvasObjectEraserLine || !renderer);

      if (!renderer) {
        renderer = new CanvasObjectEraserLine(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'rect') {
      assert(renderer instanceof CanvasObjectRect || !renderer);

      if (!renderer) {
        renderer = new CanvasObjectRect(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'image') {
      assert(renderer instanceof CanvasObjectImage || !renderer);

      if (!renderer) {
        renderer = new CanvasObjectImage(objectState, this);
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
      assert(this.bufferRenderer instanceof CanvasObjectBrushLine || !this.bufferRenderer);

      if (!this.bufferRenderer) {
        this.bufferRenderer = new CanvasObjectBrushLine(this.bufferState, this);
        this.konva.bufferGroup.add(this.bufferRenderer.konva.group);
      }

      didRender = this.bufferRenderer.update(this.bufferState, true);
    } else if (this.bufferState.type === 'eraser_line') {
      assert(this.bufferRenderer instanceof CanvasObjectEraserLine || !this.bufferRenderer);

      if (!this.bufferRenderer) {
        this.bufferRenderer = new CanvasObjectEraserLine(this.bufferState, this);
        this.konva.bufferGroup.add(this.bufferRenderer.konva.group);
      }

      didRender = this.bufferRenderer.update(this.bufferState, true);
    } else if (this.bufferState.type === 'rect') {
      assert(this.bufferRenderer instanceof CanvasObjectRect || !this.bufferRenderer);

      if (!this.bufferRenderer) {
        this.bufferRenderer = new CanvasObjectRect(this.bufferState, this);
        this.konva.bufferGroup.add(this.bufferRenderer.konva.group);
      }

      didRender = this.bufferRenderer.update(this.bufferState, true);
    } else if (this.bufferState.type === 'image') {
      assert(this.bufferRenderer instanceof CanvasObjectImage || !this.bufferRenderer);

      if (!this.bufferRenderer) {
        this.bufferRenderer = new CanvasObjectImage(this.bufferState, this);
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
      this.syncCache(true);
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
      const entityIdentifier = this.parent.entityIdentifier;
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
      const isEraserLine = renderer instanceof CanvasObjectEraserLine;
      const isImage = renderer instanceof CanvasObjectImage;
      const hasClip = renderer instanceof CanvasObjectBrushLine && renderer.state.clip;
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
  rasterize = async (options: {
    rect: Rect;
    replaceObjects?: boolean;
    attrs?: GroupConfig;
    bg?: string;
  }): Promise<ImageDTO> => {
    const { rect, replaceObjects, attrs, bg } = { replaceObjects: false, attrs: {}, ...options };
    let imageDTO: ImageDTO | null = null;
    const rasterizeArgs = { rect, attrs, bg };
    const hash = this.parent.hash(rasterizeArgs);
    const cachedImageName = this.manager.cache.imageNameCache.get(hash);

    if (cachedImageName) {
      imageDTO = await getImageDTO(cachedImageName);
      if (imageDTO) {
        this.log.trace({ rect, cachedImageName, imageDTO }, 'Using cached rasterized image');
        return imageDTO;
      }
    }

    this.log.trace({ rasterizeArgs }, 'Rasterizing entity');

    const blob = await this.getBlob(rasterizeArgs);
    if (this.manager._isDebugging) {
      previewBlob(blob, 'Rasterized entity');
    }
    imageDTO = await uploadImage({
      blob,
      fileName: `${this.id}_rasterized.png`,
      image_category: 'other',
      is_intermediate: true,
    });
    const imageObject = imageDTOToImageObject(imageDTO);
    if (replaceObjects) {
      await this.setBuffer(imageObject);
      this.commitBuffer({ pushToState: false });
    }
    this.manager.stateApi.rasterizeEntity({
      entityIdentifier: this.parent.entityIdentifier,
      imageObject,
      rect: { x: Math.round(rect.x), y: Math.round(rect.y), width: imageDTO.width, height: imageDTO.height },
      replaceObjects,
    });
    this.manager.cache.imageNameCache.set(hash, imageDTO.image_name);

    return imageDTO;
  };

  updatePreviewCanvas = debounce(() => {
    if (this.parent.transformer.$isPendingRectCalculation.get()) {
      return;
    }
    const pixelRect = this.parent.transformer.$pixelRect.get();
    if (pixelRect.width === 0 || pixelRect.height === 0) {
      return;
    }
    try {
      // TODO(psyche): This is an internal Konva method, so it may break in the future. Can we make this API public?
      const canvas = this.konva.objectGroup._getCachedSceneCanvas()._canvas as HTMLCanvasElement | undefined | null;
      if (canvas) {
        const nodeRect = this.parent.transformer.$nodeRect.get();
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

  cloneObjectGroup = (arg: { attrs?: GroupConfig } = {}): Konva.Group => {
    const { attrs } = arg;
    const clone = this.konva.objectGroup.clone();
    if (attrs) {
      clone.setAttrs(attrs);
    }
    clone.cache();
    return clone;
  };

  getCanvas = (arg: { rect?: Rect; attrs?: GroupConfig; bg?: string } = {}): HTMLCanvasElement => {
    const { rect, attrs, bg } = arg;
    const clone = this.cloneObjectGroup({ attrs });
    const canvas = konvaNodeToCanvas({ node: clone, rect, bg });
    clone.destroy();
    return canvas;
  };

  getBlob = async (arg: { rect?: Rect; attrs?: GroupConfig; bg?: string } = {}): Promise<Blob> => {
    const { rect, attrs, bg } = arg;
    const clone = this.cloneObjectGroup({ attrs });
    const blob = await konvaNodeToBlob({ node: clone, rect, bg });
    return blob;
  };

  getImageData = (arg: { rect?: Rect; attrs?: GroupConfig; bg?: string } = {}): ImageData => {
    const { rect, attrs, bg } = arg;
    const clone = this.cloneObjectGroup({ attrs });
    const imageData = konvaNodeToImageData({ node: clone, rect, bg });
    clone.destroy();
    return imageData;
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    for (const renderer of this.renderers.values()) {
      renderer.destroy();
    }
    this.renderers.clear();
  };

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
}
