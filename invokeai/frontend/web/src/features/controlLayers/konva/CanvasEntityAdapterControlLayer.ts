import type { SerializableObject } from 'common/types';
import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntityAdapterBase';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasControlLayerState, CanvasEntityIdentifier, Rect } from 'features/controlLayers/store/types';
import type { GroupConfig } from 'konva/lib/Group';
import { omit } from 'lodash-es';

export class CanvasEntityAdapterControlLayer extends CanvasEntityAdapterBase<CanvasControlLayerState> {
  static TYPE = 'control_layer_adapter';

  constructor(entityIdentifier: CanvasEntityIdentifier<'control_layer'>, manager: CanvasManager) {
    super(entityIdentifier, manager, CanvasEntityAdapterControlLayer.TYPE);
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
    if (force || this.state.withTransparencyEffect !== prevState.withTransparencyEffect) {
      this.renderer.updateTransparencyEffect();
    }
  };

  syncTransparencyEffect = () => {
    this.renderer.updateTransparencyEffect();
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
