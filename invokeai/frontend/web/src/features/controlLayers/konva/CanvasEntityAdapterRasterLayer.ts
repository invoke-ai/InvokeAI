import type { SerializableObject } from 'common/types';
import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntityAdapterBase';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasEntityIdentifier, CanvasRasterLayerState, Rect } from 'features/controlLayers/store/types';
import type { GroupConfig } from 'konva/lib/Group';
import { omit } from 'lodash-es';

export class CanvasEntityAdapterRasterLayer extends CanvasEntityAdapterBase<CanvasRasterLayerState> {
  static TYPE = 'raster_layer_adapter';

  constructor(entityIdentifier: CanvasEntityIdentifier<'raster_layer'>, manager: CanvasManager) {
    super(entityIdentifier, manager, CanvasEntityAdapterRasterLayer.TYPE);
    this.subscriptions.add(this.manager.stateApi.store.subscribe(this.sync));
    this.sync(true);
  }

  sync = (force?: boolean) => {
    const prevState = this.state;
    const state = this.getSnapshot();

    if (!state) {
      this.destroy();
      return;
    }

    this.state = state;

    if (!force && prevState === this.state) {
      return;
    }

    if (force || this.state.isEnabled !== prevState.isEnabled) {
      this.syncIsEnabled();
    }
    if (force || this.state.isLocked !== prevState.isLocked) {
      this.syncIsLocked();
    }
    if (force || this.state.objects !== prevState.objects) {
      this.syncObjects();
    }
    if (force || this.state.position !== prevState.position) {
      this.syncPosition();
    }
    if (force || this.state.opacity !== prevState.opacity) {
      this.syncOpacity();
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
    const keysToOmit: (keyof CanvasRasterLayerState)[] = ['name'];
    return omit(this.state, keysToOmit);
  };
}
