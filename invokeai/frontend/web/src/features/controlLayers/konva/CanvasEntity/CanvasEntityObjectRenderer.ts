import { $authToken } from 'app/store/nanostores/authToken';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { SyncableMap } from 'common/util/SyncableMap/SyncableMap';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectBrushLine } from 'features/controlLayers/konva/CanvasObject/CanvasObjectBrushLine';
import { CanvasObjectBrushLineWithPressure } from 'features/controlLayers/konva/CanvasObject/CanvasObjectBrushLineWithPressure';
import { CanvasObjectEraserLine } from 'features/controlLayers/konva/CanvasObject/CanvasObjectEraserLine';
import { CanvasObjectEraserLineWithPressure } from 'features/controlLayers/konva/CanvasObject/CanvasObjectEraserLineWithPressure';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import { CanvasObjectRect } from 'features/controlLayers/konva/CanvasObject/CanvasObjectRect';
import type { AnyObjectRenderer, AnyObjectState } from 'features/controlLayers/konva/CanvasObject/types';
import { LightnessToAlphaFilter } from 'features/controlLayers/konva/filters';
import { getPatternSVG } from 'features/controlLayers/konva/patterns/getPatternSVG';
import {
  getKonvaNodeDebugAttrs,
  getPrefixedId,
  konvaNodeToBlob,
  konvaNodeToCanvas,
  konvaNodeToImageData,
  previewBlob,
} from 'features/controlLayers/konva/util';
import type { Rect } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import Konva from 'konva';
import type { GroupConfig } from 'konva/lib/Group';
import { throttle } from 'lodash-es';
import type { Logger } from 'roarr';
import { getImageDTOSafe, uploadImage } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

function setFillPatternImage(shape: Konva.Shape, ...args: Parameters<typeof getPatternSVG>): HTMLImageElement {
  const imageElement = new Image();
  imageElement.onload = () => {
    shape.fillPatternImage(imageElement);
  };
  imageElement.crossOrigin = $authToken.get() ? 'use-credentials' : 'anonymous';
  imageElement.src = getPatternSVG(...args);
  return imageElement;
}

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
   * A map of object renderers, keyed by their ID.
   *
   * This map can be used with React.useSyncExternalStore to sync the object renderers with a React component.
   */
  renderers = new SyncableMap<string, AnyObjectRenderer>();

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
      compositing: null,
    };

    this.parent.konva.layer.add(this.konva.objectGroup);

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

    // The compositing rect must cover the whole stage at all times. When the stage is scaled, moved or resized, we
    // need to update the compositing rect to match the stage.
    this.subscriptions.add(
      this.manager.stage.$stageAttrs.listen((stageAttrs, oldStageAttrs) => {
        if (!this.konva.compositing) {
          return;
        }

        if (
          stageAttrs.width !== oldStageAttrs.width ||
          stageAttrs.height !== oldStageAttrs.height ||
          stageAttrs.scale !== oldStageAttrs.scale
        ) {
          this.updateCompositingRectSize();
        }

        if (stageAttrs.x !== oldStageAttrs.x || stageAttrs.y !== oldStageAttrs.y) {
          this.updateCompositingRectPosition();
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

    this.syncKonvaCache(didRender);

    return didRender;
  };

  adoptObjectRenderer = (renderer: AnyObjectRenderer) => {
    this.renderers.set(renderer.id, renderer);
    renderer.konva.group.moveTo(this.konva.objectGroup);
  };

  syncKonvaCache = (force: boolean = false) => {
    if (this.renderers.size === 0) {
      this.log.trace('Clearing object group cache');
      this.konva.objectGroup.clearCache();
      return;
    }

    // We should never cache the entity if it is not visible - it will cache as a transparent image.
    const isVisible = this.parent.konva.layer.visible();
    const isCached = this.konva.objectGroup.isCached();

    if (isVisible && (force || !isCached)) {
      this.log.trace('Caching object group');
      this.konva.objectGroup.clearCache();
      this.konva.objectGroup.cache({ pixelRatio: 1, imageSmoothingEnabled: false });
    }
  };

  updateTransparencyEffect = () => {
    if (this.parent.state.type === 'control_layer') {
      const filters = this.parent.state.withTransparencyEffect ? [LightnessToAlphaFilter] : [];
      this.konva.objectGroup.filters(filters);
    }
  };

  updateCompositingRectFill = throttle((force?: boolean) => {
    if (!force && !this.hasObjects()) {
      return;
    }

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
  }, 100);

  updateCompositingRectSize = (force?: boolean) => {
    if (!force && !this.hasObjects()) {
      return;
    }

    this.log.trace('Updating compositing rect size');

    assert(this.konva.compositing, 'Missing compositing rect');

    const scale = this.manager.stage.unscale(1);

    this.konva.compositing.rect.setAttrs({
      ...this.manager.stage.getScaledStageRect(),
      fillPatternScaleX: scale,
      fillPatternScaleY: scale,
    });
  };

  updateCompositingRectPosition = (force?: boolean) => {
    if (!force && !this.hasObjects()) {
      return;
    }

    this.log.trace('Updating compositing rect position');

    assert(this.konva.compositing, 'Missing compositing rect');

    this.konva.compositing.rect.setAttrs({
      ...this.manager.stage.getScaledStageRect(),
    });
  };

  updateOpacity = throttle(() => {
    this.log.trace('Updating opacity');

    const opacity = this.parent.state.opacity;

    if (this.konva.compositing) {
      this.konva.compositing.group.opacity(opacity);
    } else {
      this.konva.objectGroup.opacity(opacity);
    }
    this.parent.bufferRenderer.konva.group.opacity(opacity);
  }, 100);

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
    } else if (objectState.type === 'brush_line_with_pressure') {
      assert(renderer instanceof CanvasObjectBrushLineWithPressure || !renderer);

      if (!renderer) {
        renderer = new CanvasObjectBrushLineWithPressure(objectState, this);
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
    } else if (objectState.type === 'eraser_line_with_pressure') {
      assert(renderer instanceof CanvasObjectEraserLineWithPressure || !renderer);

      if (!renderer) {
        renderer = new CanvasObjectEraserLineWithPressure(objectState, this);
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
    return this.renderers.size > 0 || this.parent.bufferRenderer.hasBuffer();
  };

  /**
   * Rasterizes the parent entity, returning a promise that resolves to the image DTO.
   *
   * If the entity has a rasterization cache for the given rect, the cached image is returned. Otherwise, the entity is
   * rasterized and the image is uploaded to the server.
   *
   * The rasterization cache is reset when the entity's state changes. The buffer object is not considered part of the
   * entity state for this purpose as it is a temporary object.
   *
   * If rasterization fails for any reason, the promise will reject.
   *
   * @param options The rasterization options.
   * @param options.rect The region of the entity to rasterize.
   * @param options.replaceObjects Whether to replace the entity's objects with the rasterized image. If you just want
   * the entity's image, omit or set this to false.
   * @param options.attrs The Konva node attributes to apply to the rasterized image group. For example, you might want
   * to disable filters or set the opacity to the rasterized image.
   * @param options.bg Draws the entity on a canvas with the given background color. If omitted, the entity is drawn on
   * a transparent canvas.
   * @returns A promise that resolves to the rasterized image DTO or rejects if rasterization fails.
   */
  rasterize = async (options: {
    rect: Rect;
    replaceObjects?: boolean;
    attrs?: GroupConfig;
    bg?: string;
    ignoreCache?: boolean;
  }): Promise<ImageDTO> => {
    const rasterizingAdapter = this.manager.stateApi.$rasterizingAdapter.get();
    if (rasterizingAdapter) {
      assert(false, `Already rasterizing an entity: ${rasterizingAdapter.id}`);
    }

    const { rect, replaceObjects, attrs, bg, ignoreCache } = {
      replaceObjects: false,
      ignoreCache: false,
      attrs: {},
      ...options,
    };
    let imageDTO: ImageDTO | null = null;
    const rasterizeArgs = { rect, attrs, bg };
    const hash = this.parent.hash(rasterizeArgs);
    const cachedImageName = this.manager.cache.imageNameCache.get(hash);

    if (cachedImageName && !ignoreCache) {
      imageDTO = await getImageDTOSafe(cachedImageName);
      if (imageDTO) {
        this.log.trace({ rect, cachedImageName, imageDTO }, 'Using cached rasterized image');
        return imageDTO;
      }
    }

    this.log.trace({ rasterizeArgs }, 'Rasterizing entity');
    this.manager.stateApi.$rasterizingAdapter.set(this.parent);

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
      await this.parent.bufferRenderer.setBuffer(imageObject);
      this.parent.bufferRenderer.commitBuffer({ pushToState: false });
    }
    this.manager.stateApi.rasterizeEntity({
      entityIdentifier: this.parent.entityIdentifier,
      imageObject,
      position: { x: Math.round(rect.x), y: Math.round(rect.y) },
      replaceObjects,
    });
    this.manager.cache.imageNameCache.set(hash, imageDTO.image_name);
    this.manager.stateApi.$rasterizingAdapter.set(null);
    return imageDTO;
  };

  cloneObjectGroup = (arg: { attrs?: GroupConfig } = {}): Konva.Group => {
    const { attrs } = arg;
    const clone = this.konva.objectGroup.clone();
    if (attrs) {
      clone.setAttrs(attrs);
    }
    if (clone.hasChildren()) {
      clone.cache({ pixelRatio: 1, imageSmoothingEnabled: false });
    }
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
      konva: {
        objectGroup: getKonvaNodeDebugAttrs(this.konva.objectGroup),
      },
    };
  };
}
