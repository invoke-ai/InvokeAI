import type { SerializableObject } from 'common/types';
import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntityAdapterBase';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasControlLayerState, CanvasEntityIdentifier, Rect } from 'features/controlLayers/store/types';
import type { GroupConfig } from 'konva/lib/Group';
import { omit } from 'lodash-es';
import { assert } from 'tsafe';

export class CanvasControlLayerAdapter extends CanvasEntityAdapterBase<CanvasControlLayerState> {
  static TYPE = 'control_layer_adapter';
  private _state: CanvasControlLayerState | null = null;

  constructor(entityIdentifier: CanvasEntityIdentifier<'control_layer'>, manager: CanvasManager) {
    super(entityIdentifier, manager, CanvasControlLayerAdapter.TYPE);
  }

  get state(): CanvasControlLayerState {
    if (this._state) {
      return this._state;
    }
    const state = this.manager.stateApi.getControlLayersState().entities.find((layer) => layer.id === this.id);
    assert(state, `State not found for ${this.id}`);
    return state;
  }

  set state(state: CanvasControlLayerState) {
    const prevState = this._state;
    this._state = state;
    this.render(state, prevState);
  }

  private render = async (state: CanvasControlLayerState, prevState: CanvasControlLayerState | null): Promise<void> => {
    if (prevState && prevState === state) {
      this.log.trace('State unchanged, skipping update');
      return;
    }

    if (!prevState || state.isEnabled !== prevState.isEnabled) {
      this.log.trace('Updating visibility');
      this.konva.layer.visible(state.isEnabled);
      this.renderer.syncCache(state.isEnabled);
    }
    if (!prevState || state.isLocked !== prevState.isLocked) {
      this.transformer.syncInteractionState();
    }
    if (!prevState || state.objects !== prevState.objects) {
      const didRender = await this.renderer.render(this.state.objects);
      if (didRender) {
        this.transformer.requestRectCalculation();
      }
    }
    if (!prevState || state.position !== prevState.position) {
      this.transformer.updatePosition();
    }
    if (!prevState || state.opacity !== prevState.opacity) {
      this.renderer.updateOpacity(state.opacity);
    }
    if (!prevState || state.withTransparencyEffect !== prevState.withTransparencyEffect) {
      this.renderer.updateTransparencyEffect(state.withTransparencyEffect);
    }

    if (!prevState) {
      // First render
      this.transformer.updateBbox();
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
    const keysToOmit: (keyof CanvasControlLayerState)[] = ['name', 'controlAdapter', 'withTransparencyEffect'];
    return omit(this.state, keysToOmit);
  };
}
