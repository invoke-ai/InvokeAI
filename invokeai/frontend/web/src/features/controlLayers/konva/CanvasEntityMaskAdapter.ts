import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { CanvasEntityRenderer } from 'features/controlLayers/konva/CanvasEntityRenderer';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleABC } from 'features/controlLayers/konva/CanvasModuleABC';
import { getLastPointOfLine } from 'features/controlLayers/konva/util';
import type {
  CanvasBrushLineState,
  CanvasEntityIdentifier,
  CanvasEraserLineState,
  CanvasInpaintMaskState,
  CanvasRegionalGuidanceState,
  Coordinate,
  Rect,
} from 'features/controlLayers/store/types';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { GroupConfig } from 'konva/lib/Group';
import { get, omit } from 'lodash-es';
import type { Logger } from 'roarr';
import stableHash from 'stable-hash';

export class CanvasEntityMaskAdapter extends CanvasModuleABC {
  readonly type = 'entity_mask_adapter';

  id: string;
  path: string[];
  manager: CanvasManager;
  log: Logger;
  subscriptions = new Set<() => void>();

  state: CanvasInpaintMaskState | CanvasRegionalGuidanceState;

  transformer: CanvasEntityTransformer;
  renderer: CanvasEntityRenderer;

  isFirstRender: boolean = true;

  konva: {
    layer: Konva.Layer;
  };

  constructor(state: CanvasEntityMaskAdapter['state'], manager: CanvasEntityMaskAdapter['manager']) {
    super();
    this.id = state.id;
    this.manager = manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.debug({ state }, 'Creating mask adapter module');

    this.state = state;

    this.konva = {
      layer: new Konva.Layer({
        // We need the ID on the layer to help with building the composite initial image
        // See `getCompositeLayerStageClone()`
        id: this.id,
        name: `${this.type}:layer`,
        listening: false,
        imageSmoothingEnabled: false,
      }),
    };

    this.renderer = new CanvasEntityRenderer(this);
    this.transformer = new CanvasEntityTransformer(this);
  }

  /**
   * Get this entity's entity identifier
   */
  getEntityIdentifier = (): CanvasEntityIdentifier => {
    return getEntityIdentifier(this.state);
  };

  destroy = (): void => {
    this.log.debug('Destroying mask adapter module');

    this.transformer.destroy();
    this.renderer.destroy();
    this.konva.layer.destroy();
  };

  update = async (arg?: { state: CanvasEntityMaskAdapter['state'] }) => {
    const state = get(arg, 'state', this.state);

    const prevState = this.state;
    this.state = state;

    if (!this.isFirstRender && prevState === state && prevState.fill === state.fill) {
      this.log.trace('State unchanged, skipping update');
      return;
    }

    this.log.debug('Updating');
    const { position, objects, isEnabled, isLocked, opacity } = state;

    if (this.isFirstRender || objects !== prevState.objects) {
      await this.updateObjects({ objects });
    }
    if (this.isFirstRender || position !== prevState.position) {
      this.transformer.updatePosition({ position });
    }
    if (this.isFirstRender || opacity !== prevState.opacity) {
      this.renderer.updateOpacity(opacity);
    }
    if (this.isFirstRender || isEnabled !== prevState.isEnabled) {
      this.updateVisibility({ isEnabled });
    }
    if (this.isFirstRender || isLocked !== prevState.isLocked) {
      this.transformer.syncInteractionState();
    }
    if (this.isFirstRender || state.fill !== prevState.fill) {
      this.renderer.updateCompositingRectFill(state.fill);
    }

    if (this.isFirstRender) {
      this.renderer.updateCompositingRectSize();
    }

    if (this.isFirstRender) {
      this.transformer.updateBbox();
    }

    this.isFirstRender = false;
  };

  updateObjects = async (arg?: { objects: CanvasInpaintMaskState['objects'] }) => {
    this.log.trace('Updating objects');

    const objects = get(arg, 'objects', this.state.objects);

    const didUpdate = await this.renderer.render(objects);

    if (didUpdate) {
      this.transformer.requestRectCalculation();
    }
  };

  updateVisibility = (arg?: { isEnabled: boolean }) => {
    this.log.trace('Updating visibility');
    const isEnabled = get(arg, 'isEnabled', this.state.isEnabled);
    this.konva.layer.visible(isEnabled);
  };

  getLastPointOfLastLine = (type: CanvasBrushLineState['type'] | CanvasEraserLineState['type']): Coordinate | null => {
    const lastObject = this.state.objects[this.state.objects.length - 1];
    if (!lastObject) {
      return null;
    }

    if (lastObject.type === type) {
      return getLastPointOfLine(lastObject.points);
    }

    return null;
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      state: deepClone(this.state),
    };
  };

  getHashableState = (): SerializableObject => {
    const keysToOmit: (keyof CanvasEntityMaskAdapter['state'])[] = ['fill', 'name', 'opacity'];
    return omit(this.state, keysToOmit);
  };

  hash = (extra?: SerializableObject): string => {
    const arg = {
      state: this.getHashableState(),
      extra,
    };
    return stableHash(arg);
  };

  getCanvas = (rect?: Rect): HTMLCanvasElement => {
    // TODO(psyche): Cache this?
    // The opacity may have been changed in response to user selecting a different entity category, and the mask regions
    // should be fully opaque - set opacity to 1 before rendering the canvas
    const attrs: GroupConfig = { opacity: 1 };
    const canvas = this.renderer.getCanvas(rect, attrs);
    return canvas;
  };

  getLoggingContext = () => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
