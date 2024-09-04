import type { SerializableObject } from 'common/types';
import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntityAdapterBase';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasEntityIdentifier, CanvasInpaintMaskState, Rect } from 'features/controlLayers/store/types';
import type { GroupConfig } from 'konva/lib/Group';
import { omit } from 'lodash-es';
import { assert } from 'tsafe';

export class CanvasInpaintMaskAdapter extends CanvasEntityAdapterBase<CanvasInpaintMaskState> {
  static TYPE = 'inpaint_mask_adapter';

  /**
   * The last known state of the entity.
   */
  private _state: CanvasInpaintMaskState | null = null;

  constructor(entityIdentifier: CanvasEntityIdentifier<'inpaint_mask'>, manager: CanvasManager) {
    super(entityIdentifier, manager, CanvasInpaintMaskAdapter.TYPE);
  }

  get state(): CanvasInpaintMaskState {
    if (this._state) {
      return this._state;
    }
    const state = this.manager.stateApi.getInpaintMasksState().entities.find((layer) => layer.id === this.id);
    assert(state, `State not found for ${this.id}`);
    return state;
  }

  set state(state: CanvasInpaintMaskState) {
    this._state = state;
  }

  update = async (state: CanvasInpaintMaskState) => {
    const prevState = this.state;
    this.state = state;

    if (prevState && prevState === state) {
      this.log.trace('State unchanged, skipping update');
      return;
    }

    if (!prevState || state.isEnabled !== prevState.isEnabled) {
      this.log.trace('Updating visibility');
      this.konva.layer.visible(state.isEnabled);
      this.renderer.syncCache(state.isEnabled);
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
    if (!prevState || state.isLocked !== prevState.isLocked) {
      this.transformer.syncInteractionState();
    }
    if (!prevState || state.fill !== prevState.fill) {
      this.renderer.updateCompositingRectFill(state.fill);
    }

    if (!prevState) {
      this.renderer.updateCompositingRectSize();
    }

    if (!prevState) {
      this.transformer.updateBbox();
    }
  };

  getHashableState = (): SerializableObject => {
    const keysToOmit: (keyof CanvasInpaintMaskState)[] = ['fill', 'name', 'opacity'];
    return omit(this.state, keysToOmit);
  };

  getCanvas = (rect?: Rect): HTMLCanvasElement => {
    // The opacity may have been changed in response to user selecting a different entity category, and the mask regions
    // should be fully opaque - set opacity to 1 before rendering the canvas
    const attrs: GroupConfig = { opacity: 1 };
    const canvas = this.renderer.getCanvas(rect, attrs);
    return canvas;
  };
}
