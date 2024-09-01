import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { CanvasEntityRenderer } from 'features/controlLayers/konva/CanvasEntityRenderer';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getLastPointOfLine } from 'features/controlLayers/konva/util';
import type {
  CanvasBrushLineState,
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasEraserLineState,
  CanvasRasterLayerState,
  Coordinate,
  Rect,
} from 'features/controlLayers/store/types';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { GroupConfig } from 'konva/lib/Group';
import { get, omit } from 'lodash-es';
import type { Logger } from 'roarr';
import stableHash from 'stable-hash';
import { assert } from 'tsafe';

/**
 * Handles the rendering for a single raster or control layer entity.
 *
 * This module has two main components:
 * - A transformer, which handles the positioning and interaction state of the layer
 * - A renderer, which handles the rendering of the layer's objects
 *
 * The canvas rendering module interacts with this module to coordinate the rendering of all raster and control layers.
 */
export class CanvasEntityLayerAdapter extends CanvasModuleBase {
  readonly type = 'entity_layer_adapter';

  id: string;
  path: string[];
  manager: CanvasManager;
  parent: CanvasManager;
  log: Logger;

  /**
   * The last known state of the entity.
   */
  state: CanvasRasterLayerState | CanvasControlLayerState;

  /**
   * The Konva nodes that make up the entity layer:
   * - A layer to hold the everything
   *
   * Note that the transformer and object renderer have their own Konva nodes, but they are not stored here.
   */
  konva: {
    layer: Konva.Layer;
  };

  /**
   * The transformer for this entity layer.
   */
  transformer: CanvasEntityTransformer;

  /**
   * The renderer for this entity layer.
   */
  renderer: CanvasEntityRenderer;

  /**
   * Whether this is the first render of the entity layer.
   */
  isFirstRender: boolean = true;

  constructor(state: CanvasEntityLayerAdapter['state'], manager: CanvasEntityLayerAdapter['manager']) {
    super();
    this.id = state.id;
    this.manager = manager;
    this.parent = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

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

  update = async (arg?: { state: CanvasEntityLayerAdapter['state'] }): Promise<void> => {
    const state = get(arg, 'state', this.state);

    const prevState = this.state;
    this.state = state;

    if (!this.isFirstRender && prevState === state) {
      this.log.trace('State unchanged, skipping update');
      return;
    }

    this.log.debug('Updating');
    const { position, objects, opacity, isEnabled, isLocked } = state;

    if (this.isFirstRender || isEnabled !== prevState.isEnabled) {
      this.updateVisibility({ isEnabled });
    }
    if (this.isFirstRender || isLocked !== prevState.isLocked) {
      this.transformer.syncInteractionState();
    }
    if (this.isFirstRender || objects !== prevState.objects) {
      await this.updateObjects({ objects });
    }
    if (this.isFirstRender || position !== prevState.position) {
      this.transformer.updatePosition({ position });
    }
    if (this.isFirstRender || opacity !== prevState.opacity) {
      this.renderer.updateOpacity(opacity);
    }

    if (state.type === 'control_layer' && prevState.type === 'control_layer') {
      if (this.isFirstRender || state.withTransparencyEffect !== prevState.withTransparencyEffect) {
        this.renderer.updateTransparencyEffect(state.withTransparencyEffect);
      }
    }

    if (this.isFirstRender) {
      this.transformer.updateBbox();
    }

    this.isFirstRender = false;
  };

  updateVisibility = (arg?: { isEnabled: boolean }) => {
    this.log.trace('Updating visibility');
    const isEnabled = get(arg, 'isEnabled', this.state.isEnabled);
    this.konva.layer.visible(isEnabled);
    this.renderer.syncCache(isEnabled);
  };

  updateObjects = async (arg?: { objects: CanvasRasterLayerState['objects'] }) => {
    this.log.trace('Updating objects');

    const objects = get(arg, 'objects', this.state.objects);

    const didUpdate = await this.renderer.render(objects);

    if (didUpdate) {
      this.transformer.requestRectCalculation();
    }
  };

  getCanvas = (rect?: Rect): HTMLCanvasElement => {
    this.log.trace({ rect }, 'Getting canvas');
    // The opacity may have been changed in response to user selecting a different entity category, so we must restore
    // the original opacity before rendering the canvas
    const attrs: GroupConfig = { opacity: this.state.opacity };
    const canvas = this.renderer.getCanvas(rect, attrs);
    return canvas;
  };

  getHashableState = (): SerializableObject => {
    if (this.state.type === 'control_layer') {
      const keysToOmit: (keyof CanvasControlLayerState)[] = ['name', 'controlAdapter', 'withTransparencyEffect'];
      return omit(this.state, keysToOmit);
    } else if (this.state.type === 'raster_layer') {
      const keysToOmit: (keyof CanvasRasterLayerState)[] = ['name'];
      return omit(this.state, keysToOmit);
    } else {
      assert(false, 'Unexpected layer type');
    }
  };

  hash = (extra?: SerializableObject): string => {
    const arg = {
      state: this.getHashableState(),
      extra,
    };
    return stableHash(arg);
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

  destroy = (): void => {
    this.log.debug('Destroying module');
    this.renderer.destroy();
    this.transformer.destroy();
    this.konva.layer.destroy();
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      state: deepClone(this.state),
      transformer: this.transformer.repr(),
      renderer: this.renderer.repr(),
    };
  };
}
