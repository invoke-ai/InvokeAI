import type { JSONObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { CanvasBrushLineRenderer } from 'features/controlLayers/konva/CanvasBrushLine';
import { CanvasEraserLineRenderer } from 'features/controlLayers/konva/CanvasEraserLine';
import { CanvasImageRenderer } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasRectRenderer } from 'features/controlLayers/konva/CanvasRect';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type {
  CanvasBrushLineState,
  CanvasEraserLineState,
  CanvasImageState,
  CanvasRectState,
} from 'features/controlLayers/store/types';
import type { Logger } from 'roarr';
import { assert } from 'tsafe';

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
  static TYPE = 'object_renderer';

  id: string;
  parent: CanvasLayer;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: (extra?: JSONObject) => JSONObject;

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

  constructor(parent: CanvasLayer) {
    this.id = getPrefixedId(CanvasObjectRenderer.TYPE);
    this.parent = parent;
    this.manager = parent.manager;
    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.trace('Creating object renderer');
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

    return didRender;
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
        this.parent.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'eraser_line') {
      assert(renderer instanceof CanvasEraserLineRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasEraserLineRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.parent.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'rect') {
      assert(renderer instanceof CanvasRectRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasRectRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.parent.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force || isFirstRender);
    } else if (objectState.type === 'image') {
      assert(renderer instanceof CanvasImageRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasImageRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.parent.konva.objectGroup.add(renderer.konva.group);
      }
      didRender = await renderer.update(objectState, force || isFirstRender);
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
    this.buffer = objectState;
    return await this.renderObject(this.buffer, true);
  };

  /**
   * Clears the buffer object state.
   */
  clearBuffer = () => {
    this.buffer = null;
  };

  /**
   * Commits the current buffer object, pushing the buffer object state back to the application state.
   */
  commitBuffer = () => {
    if (!this.buffer) {
      this.log.warn('No buffer object to commit');
      return;
    }

    // We need to give the objects a fresh ID else they will be considered the same object when they are re-rendered as
    // a non-buffer object, and we won't trigger things like bbox calculation
    this.buffer.id = getPrefixedId(this.buffer.type);

    if (this.buffer.type === 'brush_line') {
      this.manager.stateApi.onBrushLineAdded({ id: this.parent.id, brushLine: this.buffer }, 'layer');
    } else if (this.buffer.type === 'eraser_line') {
      this.manager.stateApi.onEraserLineAdded({ id: this.parent.id, eraserLine: this.buffer }, 'layer');
    } else if (this.buffer.type === 'rect') {
      this.manager.stateApi.onRectShapeAdded({ id: this.parent.id, rectShape: this.buffer }, 'layer');
    } else {
      this.log.warn({ buffer: this.buffer }, 'Invalid buffer object type');
    }

    this.buffer = null;
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

  /**
   * Destroys this renderer and all of its object renderers.
   */
  destroy = () => {
    this.log.trace('Destroying object renderer');
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
      type: CanvasObjectRenderer.TYPE,
      parent: this.parent.id,
      renderers: Array.from(this.renderers.values()).map((renderer) => renderer.repr()),
      buffer: deepClone(this.buffer),
    };
  };
}
