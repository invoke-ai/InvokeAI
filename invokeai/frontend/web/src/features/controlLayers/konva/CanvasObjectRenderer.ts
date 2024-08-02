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

type AnyObjectRenderer = CanvasBrushLineRenderer | CanvasEraserLineRenderer | CanvasRectRenderer | CanvasImageRenderer;
type AnyObjectState = CanvasBrushLineState | CanvasEraserLineState | CanvasImageState | CanvasRectState;

export class CanvasObjectRenderer {
  static TYPE = 'object_renderer';
  static OBJECT_GROUP_NAME = `${CanvasObjectRenderer.TYPE}_group`;

  id: string;
  parent: CanvasLayer;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: (extra?: JSONObject) => JSONObject;

  isFirstRender: boolean = true;
  isRendering: boolean = false;
  buffer: AnyObjectState | null = null;
  renderers: Map<string, AnyObjectRenderer> = new Map();

  constructor(parent: CanvasLayer) {
    this.id = getPrefixedId(CanvasObjectRenderer.TYPE);
    this.parent = parent;
    this.manager = parent.manager;
    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.trace('Creating object renderer');
  }

  render = async (objectStates: AnyObjectState[]): Promise<boolean> => {
    this.isRendering = true;
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

    this.isRendering = false;
    this.isFirstRender = false;

    return didRender;
  };

  renderObject = async (objectState: AnyObjectState, force?: boolean): Promise<boolean> => {
    let didRender = false;

    if (objectState.type === 'brush_line') {
      let renderer = this.renderers.get(objectState.id);
      assert(renderer instanceof CanvasBrushLineRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasBrushLineRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.parent.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force);
    } else if (objectState.type === 'eraser_line') {
      let renderer = this.renderers.get(objectState.id);
      assert(renderer instanceof CanvasEraserLineRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasEraserLineRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.parent.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force);
    } else if (objectState.type === 'rect') {
      let renderer = this.renderers.get(objectState.id);
      assert(renderer instanceof CanvasRectRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasRectRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.parent.konva.objectGroup.add(renderer.konva.group);
      }

      didRender = renderer.update(objectState, force);
    } else if (objectState.type === 'image') {
      let renderer = this.renderers.get(objectState.id);
      assert(renderer instanceof CanvasImageRenderer || renderer === undefined);

      if (!renderer) {
        renderer = new CanvasImageRenderer(objectState, this);
        this.renderers.set(renderer.id, renderer);
        this.parent.konva.objectGroup.add(renderer.konva.group);
      }
      didRender = await renderer.update(objectState, force);
    }

    this.isFirstRender = false;
    return didRender;
  };

  hasBuffer = (): boolean => {
    return this.buffer !== null;
  };

  setBuffer = async (objectState: AnyObjectState): Promise<boolean> => {
    this.buffer = objectState;
    return await this.renderObject(this.buffer, true);
  };

  clearBuffer = () => {
    this.buffer = null;
  };

  commitBuffer = () => {
    if (!this.buffer) {
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

  hasObjects = (): boolean => {
    return this.renderers.size > 0 || this.buffer !== null;
  };

  hideAll = (except: string[]) => {
    for (const renderer of this.renderers.values()) {
      if (!except.includes(renderer.id)) {
        renderer.setVisibility(false);
      }
    }
  };

  destroy = () => {
    this.log.trace('Destroying object renderer');
    for (const renderer of this.renderers.values()) {
      renderer.destroy();
    }
    this.renderers.clear();
  };

  repr = () => {
    return {
      id: this.id,
      type: CanvasObjectRenderer.TYPE,
      parent: this.parent.id,
      renderers: Array.from(this.renderers.values()).map((renderer) => renderer.repr()),
      buffer: deepClone(this.buffer),
      isFirstRender: this.isFirstRender,
      isRendering: this.isRendering,
    };
  };
}
