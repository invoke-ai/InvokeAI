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

  it('publishes the requested layer and section', () => {
    requestLayerProperties('control-1', 'filter');

    expect(getLayerPropertiesRequest()).toMatchObject({ layerId: 'control-1', section: 'filter' });
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
});
