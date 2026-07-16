import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createControlLayer, createEmptyPaintLayer } from './layerOps';
import {
  clearLayerPropertiesRequest,
  getLayerPropertiesRequest,
  isLayerPropertiesGroupRequested,
  layerPropertiesRequestStore,
  requestLayerProperties,
} from './layerPropertiesRequestStore';

describe('layerPropertiesRequestStore', () => {
  beforeEach(() => {
    clearLayerPropertiesRequest();
  });

  it.each(['filter', 'adjustments'] as const)('publishes and token-safely clears a %s request', (section) => {
    requestLayerProperties('layer-1', section);
    const first = getLayerPropertiesRequest();
    requestLayerProperties('layer-2', section);
    const second = getLayerPropertiesRequest();

    expect(first).toMatchObject({ layerId: 'layer-1', section });
    expect(second).toMatchObject({ layerId: 'layer-2', section });
    clearLayerPropertiesRequest(first?.token);
    expect(getLayerPropertiesRequest()).toEqual(second);
    clearLayerPropertiesRequest(second?.token);
    expect(getLayerPropertiesRequest()).toBeNull();
  });

  it('publishes a fresh token for repeated requests and notifies subscribers', () => {
    const listener = vi.fn();
    const unsubscribe = layerPropertiesRequestStore.subscribe(listener);

    requestLayerProperties('control-1', 'filter');
    const first = getLayerPropertiesRequest();
    requestLayerProperties('control-1', 'filter');
    const second = getLayerPropertiesRequest();

    expect(second?.token).toBeGreaterThan(first?.token ?? 0);
    expect(listener).toHaveBeenCalledTimes(2);
    unsubscribe();
  });

  it('only clears the request matching the supplied token', () => {
    requestLayerProperties('control-1', 'filter');
    const first = getLayerPropertiesRequest();
    requestLayerProperties('control-2', 'filter');
    const second = getLayerPropertiesRequest();

    clearLayerPropertiesRequest(first?.token);
    expect(getLayerPropertiesRequest()).toEqual(second);
    clearLayerPropertiesRequest(second?.token);
    expect(getLayerPropertiesRequest()).toBeNull();
  });

  it('identifies the collapsed layer group that must mount to consume the request', () => {
    const control = createControlLayer('Control', 'control-1');
    requestLayerProperties(control.id, 'filter');
    const request = getLayerPropertiesRequest();

    expect(isLayerPropertiesGroupRequested(request, [control])).toBe(true);
    expect(isLayerPropertiesGroupRequested(request, [createEmptyPaintLayer('Raster', 'raster-1')])).toBe(false);
  });
});
