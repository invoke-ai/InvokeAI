import { omit } from 'es-toolkit/compat';
import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterBase';
import { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityTransformer';
import { CanvasEntityVectorLayerRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityVectorLayerRenderer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasEntityIdentifier, CanvasVectorLayerState, Rect } from 'features/controlLayers/store/types';
import type { GroupConfig } from 'konva/lib/Group';
import type { JsonObject } from 'type-fest';

export class CanvasEntityAdapterVectorLayer extends CanvasEntityAdapterBase<
  CanvasVectorLayerState,
  'vector_layer_adapter'
> {
  renderer: CanvasEntityVectorLayerRenderer;
  bufferRenderer: CanvasEntityBufferObjectRenderer;
  transformer: CanvasEntityTransformer;
  filterer = undefined;
  segmentAnything = undefined;

  constructor(entityIdentifier: CanvasEntityIdentifier<'vector_layer'>, manager: CanvasManager) {
    super(entityIdentifier, manager, 'vector_layer_adapter');

    this.renderer = new CanvasEntityVectorLayerRenderer(this);
    this.bufferRenderer = new CanvasEntityBufferObjectRenderer(this);
    this.transformer = new CanvasEntityTransformer(this);

    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(this.selectState, this.sync));
  }

  sync = async (state: CanvasVectorLayerState | undefined, prevState: CanvasVectorLayerState | undefined) => {
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
    if (!prevState || this.state.paths !== prevState.paths) {
      await this.syncPaths();
    }
    if (!prevState || this.state.position !== prevState.position) {
      this.syncPosition();
    }
    if (!prevState || this.state.opacity !== prevState.opacity) {
      this.syncOpacity();
    }
  };

  private syncPaths = async () => {
    this.$isEmpty.set(this.state.paths.length === 0);
    await this.renderer.render();
    this.transformer.requestRectCalculation();
    this.transformer.syncInteractionState();
  };

  getHashableState = (): JsonObject => {
    const keysToOmit: (keyof CanvasVectorLayerState)[] = ['name', 'isLocked'];
    return omit(this.state, keysToOmit);
  };

  getCanvas = (rect?: Rect): HTMLCanvasElement => {
    const attrs: GroupConfig = { opacity: this.state.opacity };
    return this.renderer.getCanvas({ rect, attrs });
  };
}
