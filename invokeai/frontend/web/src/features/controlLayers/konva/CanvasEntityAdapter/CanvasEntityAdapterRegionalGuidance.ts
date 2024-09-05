import type { SerializableObject } from 'common/types';
import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntityAdapter/CanvasEntityAdapterBase';
import { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntityObjectRenderer';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasEntityIdentifier, CanvasRegionalGuidanceState, Rect } from 'features/controlLayers/store/types';
import type { GroupConfig } from 'konva/lib/Group';
import { omit } from 'lodash-es';

export class CanvasEntityAdapterRegionalGuidance extends CanvasEntityAdapterBase<CanvasRegionalGuidanceState> {
  static TYPE = 'regional_guidance_adapter';

  transformer: CanvasEntityTransformer;
  renderer: CanvasEntityObjectRenderer;

  constructor(entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>, manager: CanvasManager) {
    super(entityIdentifier, manager, CanvasEntityAdapterRegionalGuidance.TYPE);

    this.transformer = new CanvasEntityTransformer(this);
    this.renderer = new CanvasEntityObjectRenderer(this);

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
    if (force || this.state.fill !== prevState.fill) {
      this.syncCompositingRectFill();
    }
    if (force) {
      this.syncCompositingRectSize();
    }
  };

  syncCompositingRectSize = () => {
    this.renderer.updateCompositingRectSize();
  };

  syncCompositingRectFill = () => {
    this.renderer.updateCompositingRectFill();
  };

  getHashableState = (): SerializableObject => {
    const keysToOmit: (keyof CanvasRegionalGuidanceState)[] = ['fill', 'name', 'opacity'];
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
