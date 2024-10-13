import type { SerializableObject } from 'common/types';
import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterBase';
import { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import { CanvasEntityFilterer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityFilterer';
import { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasEntityIdentifier, CanvasRasterLayerState, Rect } from 'features/controlLayers/store/types';
import type { GroupConfig } from 'konva/lib/Group';
import { omit } from 'lodash-es';

export class CanvasEntityAdapterRasterLayer extends CanvasEntityAdapterBase<
  CanvasRasterLayerState,
  'raster_layer_adapter'
> {
  renderer: CanvasEntityObjectRenderer;
  bufferRenderer: CanvasEntityBufferObjectRenderer;
  transformer: CanvasEntityTransformer;
  filterer: CanvasEntityFilterer;

  constructor(entityIdentifier: CanvasEntityIdentifier<'raster_layer'>, manager: CanvasManager) {
    super(entityIdentifier, manager, 'raster_layer_adapter');

    this.renderer = new CanvasEntityObjectRenderer(this);
    this.bufferRenderer = new CanvasEntityBufferObjectRenderer(this);
    this.transformer = new CanvasEntityTransformer(this);
    this.filterer = new CanvasEntityFilterer(this);

    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(this.selectState, this.sync));
  }

  sync = async (state: CanvasRasterLayerState | undefined, prevState: CanvasRasterLayerState | undefined) => {
    if (!state) {
      this.destroy();
      return;
    }

    this.state = state;

    if (prevState && prevState === this.state) {
      return;
    }

    if (!prevState || this.state.isEnabled !== prevState.isEnabled) {
      this.syncIsEnabled();
    }
    if (!prevState || this.state.isLocked !== prevState.isLocked) {
      this.syncIsLocked();
    }
    if (!prevState || this.state.objects !== prevState.objects) {
      await this.syncObjects();
    }
    if (!prevState || this.state.position !== prevState.position) {
      this.syncPosition();
    }
    if (!prevState || this.state.opacity !== prevState.opacity) {
      this.syncOpacity();
    }
  };

  getCanvas = (rect?: Rect): HTMLCanvasElement => {
    this.log.trace({ rect }, 'Getting canvas');
    // The opacity may have been changed in response to user selecting a different entity category, so we must restore
    // the original opacity before rendering the canvas
    const attrs: GroupConfig = { opacity: this.state.opacity, filters: [] };
    const canvas = this.renderer.getCanvas({ rect, attrs });
    return canvas;
  };

  getHashableState = (): SerializableObject => {
    const keysToOmit: (keyof CanvasRasterLayerState)[] = ['name'];
    return omit(this.state, keysToOmit);
  };
}
