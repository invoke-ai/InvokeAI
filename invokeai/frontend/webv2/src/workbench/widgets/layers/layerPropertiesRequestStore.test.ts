import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  clearLayerPropertiesRequest,
  getLayerPropertiesRequest,
  layerPropertiesRequestStore,
  requestLayerProperties,
} from './layerPropertiesRequestStore';

describe('layerPropertiesRequestStore', () => {
  beforeEach(() => {
    clearLayerPropertiesRequest();
  });

  it('publishes and token-safely clears a request', () => {
    requestLayerProperties('layer-1');
    const first = getLayerPropertiesRequest();
    requestLayerProperties('layer-2');
    const second = getLayerPropertiesRequest();

    expect(first).toMatchObject({ layerId: 'layer-1' });
    expect(second).toMatchObject({ layerId: 'layer-2' });
    clearLayerPropertiesRequest(first?.token);
    expect(getLayerPropertiesRequest()).toEqual(second);
    clearLayerPropertiesRequest(second?.token);
    expect(getLayerPropertiesRequest()).toBeNull();
  });

  it('publishes a fresh token for repeated requests and notifies subscribers', () => {
    const listener = vi.fn();
    const unsubscribe = layerPropertiesRequestStore.subscribe(listener);

    requestLayerProperties('control-1');
    const first = getLayerPropertiesRequest();
    requestLayerProperties('control-1');
    const second = getLayerPropertiesRequest();

    expect(second?.token).toBeGreaterThan(first?.token ?? 0);
    expect(listener).toHaveBeenCalledTimes(2);
    unsubscribe();
  });

  it('only clears the request matching the supplied token', () => {
    requestLayerProperties('control-1');
    const first = getLayerPropertiesRequest();
    requestLayerProperties('control-2');
    const second = getLayerPropertiesRequest();

    clearLayerPropertiesRequest(first?.token);
    expect(getLayerPropertiesRequest()).toEqual(second);
    clearLayerPropertiesRequest(second?.token);
    expect(getLayerPropertiesRequest()).toBeNull();
  });
});
