import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterBase';
import { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasEntityIdentifier, CanvasRegionalGuidanceState, Rect } from 'features/controlLayers/store/types';
import type { GroupConfig } from 'konva/lib/Group';
import { omit } from 'lodash-es';
import type { JsonObject } from 'type-fest';

export class CanvasEntityAdapterRegionalGuidance extends CanvasEntityAdapterBase<
  CanvasRegionalGuidanceState,
  'regional_guidance_adapter'
> {
  renderer: CanvasEntityObjectRenderer;
  bufferRenderer: CanvasEntityBufferObjectRenderer;
  transformer: CanvasEntityTransformer;
  filterer = undefined;
  segmentAnything = undefined;

  constructor(entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>, manager: CanvasManager) {
    super(entityIdentifier, manager, 'regional_guidance_adapter');

    this.renderer = new CanvasEntityObjectRenderer(this);
    this.bufferRenderer = new CanvasEntityBufferObjectRenderer(this);
    this.transformer = new CanvasEntityTransformer(this);

    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(this.selectState, this.sync));
  }

  sync = async (state: CanvasRegionalGuidanceState | undefined, prevState: CanvasRegionalGuidanceState | undefined) => {
    if (!state) {
      this.destroy();
      return;
    }

    this.state = state;

    if (prevState && prevState === this.state) {
      return;
    }

    // If prevState is undefined, this is the first render. Some logic is only needed on the first render, or required
    // on first render.

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
    if (!prevState || this.state.fill !== prevState.fill) {
      // On first render, or when the fill changes, we must force the update
      this.renderer.updateCompositingRectFill(true);
    }
    if (!prevState || this.state.objects !== prevState.objects) {
      // On first render, or when the objects change, we must force the update
      this.renderer.updateCompositingRectSize(true);
      this.renderer.updateCompositingRectPosition(true);
    }
  };

  getHashableState = (): JsonObject => {
    const keysToOmit: (keyof CanvasRegionalGuidanceState)[] = [
      'fill',
      'name',
      'opacity',
      'isLocked',
      'autoNegative',
      'positivePrompt',
      'negativePrompt',
      'referenceImages',
    ];
    return omit(this.state, keysToOmit);
  };

  getCanvas = (rect?: Rect): HTMLCanvasElement => {
    // The opacity may have been changed in response to user selecting a different entity category, and the mask regions
    // should be fully opaque - set opacity to 1 before rendering the canvas
    const attrs: GroupConfig = { opacity: 1, filters: [] };
    const canvas = this.renderer.getCanvas({ rect, attrs });
    return canvas;
  };
}
