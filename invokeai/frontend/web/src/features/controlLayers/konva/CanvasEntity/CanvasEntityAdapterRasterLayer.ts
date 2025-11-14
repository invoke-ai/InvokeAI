import { omit, throttle } from 'es-toolkit/compat';
import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterBase';
import { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import { CanvasEntityFilterer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityFilterer';
import { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasSegmentAnythingModule } from 'features/controlLayers/konva/CanvasSegmentAnythingModule';
import { AdjustmentsCurvesFilter, AdjustmentsSimpleFilter, buildCurveLUT } from 'features/controlLayers/konva/filters';
import type { CanvasEntityIdentifier, CanvasRasterLayerState, Rect } from 'features/controlLayers/store/types';
import type { GroupConfig } from 'konva/lib/Group';
import type { JsonObject } from 'type-fest';

export class CanvasEntityAdapterRasterLayer extends CanvasEntityAdapterBase<
  CanvasRasterLayerState,
  'raster_layer_adapter'
> {
  renderer: CanvasEntityObjectRenderer;
  bufferRenderer: CanvasEntityBufferObjectRenderer;
  transformer: CanvasEntityTransformer;
  filterer: CanvasEntityFilterer;
  segmentAnything: CanvasSegmentAnythingModule;

  constructor(entityIdentifier: CanvasEntityIdentifier<'raster_layer'>, manager: CanvasManager) {
    super(entityIdentifier, manager, 'raster_layer_adapter');

    this.renderer = new CanvasEntityObjectRenderer(this);
    this.bufferRenderer = new CanvasEntityBufferObjectRenderer(this);
    this.transformer = new CanvasEntityTransformer(this);
    this.filterer = new CanvasEntityFilterer(this);
    this.segmentAnything = new CanvasSegmentAnythingModule(this);

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
    if (!prevState || this.state.globalCompositeOperation !== prevState.globalCompositeOperation) {
      this.syncGlobalCompositeOperation();
    }

    // Apply per-layer adjustments as a Konva filter
    if (!prevState || this.haveAdjustmentsChanged(prevState, this.state)) {
      this.syncAdjustmentsFilter();
    }
  };

  getCanvas = (rect?: Rect): HTMLCanvasElement => {
    this.log.trace({ rect }, 'Getting canvas');
    // The opacity may have been changed in response to user selecting a different entity category, so we must restore
    // the original opacity before rendering the canvas
    const attrs: GroupConfig = { opacity: this.state.opacity };
    const canvas = this.renderer.getCanvas({ rect, attrs });
    return canvas;
  };

  getHashableState = (): JsonObject => {
    const keysToOmit: (keyof CanvasRasterLayerState)[] = ['name', 'isLocked'];
    return omit(this.state, keysToOmit);
  };

  private syncAdjustmentsFilter = () => {
    const a = this.state.adjustments;
    const apply = !!a && a.enabled;
    // The filter operates on the renderer's object group; we can set filters at the group level via renderer
    const group = this.renderer.konva.objectGroup;
    if (apply) {
      const filters = group.filters() ?? [];
      let nextFilters = filters.filter((f) => f !== AdjustmentsSimpleFilter && f !== AdjustmentsCurvesFilter);
      if (a.mode === 'simple') {
        group.setAttr('adjustmentsSimple', a.simple);
        group.setAttr('adjustmentsCurves', null);
        nextFilters = [...nextFilters, AdjustmentsSimpleFilter];
      } else {
        // Build LUTs and set curves attr
        const master = buildCurveLUT(a.curves.master);
        const r = buildCurveLUT(a.curves.r);
        const g = buildCurveLUT(a.curves.g);
        const b = buildCurveLUT(a.curves.b);
        group.setAttr('adjustmentsCurves', { master, r, g, b });
        group.setAttr('adjustmentsSimple', null);
        nextFilters = [...nextFilters, AdjustmentsCurvesFilter];
      }
      group.filters(nextFilters);
      this._throttledCacheRefresh();
    } else {
      // Remove our filter if present
      const filters = (group.filters() ?? []).filter(
        (f) => f !== AdjustmentsSimpleFilter && f !== AdjustmentsCurvesFilter
      );
      group.filters(filters);
      group.setAttr('adjustmentsSimple', null);
      group.setAttr('adjustmentsCurves', null);
      this._throttledCacheRefresh();
    }
  };

  private _throttledCacheRefresh = throttle(() => this.renderer.syncKonvaCache(true), 50);

  private haveAdjustmentsChanged = (prevState: CanvasRasterLayerState, currState: CanvasRasterLayerState): boolean => {
    const pa = prevState.adjustments;
    const ca = currState.adjustments;
    if (pa === ca) {
      return false;
    }
    if (!pa || !ca) {
      return true;
    }
    if (pa.enabled !== ca.enabled) {
      return true;
    }
    if (pa.mode !== ca.mode) {
      return true;
    }
    // simple params
    const ps = pa.simple;
    const cs = ca.simple;
    if (
      ps.brightness !== cs.brightness ||
      ps.contrast !== cs.contrast ||
      ps.saturation !== cs.saturation ||
      ps.temperature !== cs.temperature ||
      ps.tint !== cs.tint ||
      ps.sharpness !== cs.sharpness
    ) {
      return true;
    }
    // curves params
    const pc = pa.curves;
    const cc = ca.curves;
    if (pc !== cc) {
      return true;
    }
    return false;
  };

  private syncGlobalCompositeOperation = () => {
    this.log.trace('Syncing globalCompositeOperation');
    const operation = this.state.globalCompositeOperation ?? 'source-over';

    // Map globalCompositeOperation to CSS mix-blend-mode for live preview
    const mixBlendModeMap: Record<string, string> = {
      'source-over': 'normal', // this one is why we need the map
      multiply: 'multiply',
      screen: 'screen',
      overlay: 'overlay',
      darken: 'darken',
      lighten: 'lighten',
      'color-dodge': 'color-dodge',
      'color-burn': 'color-burn',
      'hard-light': 'hard-light',
      'soft-light': 'soft-light',
      difference: 'difference',
      exclusion: 'exclusion',
      hue: 'hue',
      saturation: 'saturation',
      color: 'color',
      luminosity: 'luminosity',
    };

    const mixBlendMode = mixBlendModeMap[operation] || 'normal';

    const canvasElement = this.konva.layer.getCanvas()._canvas as HTMLCanvasElement | undefined;
    if (canvasElement) {
      canvasElement.style.mixBlendMode = mixBlendMode;
    }
  };
}
