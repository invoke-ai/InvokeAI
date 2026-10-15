import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

export interface LayerPropertiesRequest {
  layerId: string;
  token: number;
}

export const layerPropertiesRequestStore = createExternalStore<{ request: LayerPropertiesRequest | null }>({
  request: null,
});

let nextToken = 1;

registerAccountOwnedResource({
  clear: () => {
    layerPropertiesRequestStore.setSnapshot({ request: null });
  },
  name: 'layer-properties-requests',
});

export const requestLayerProperties = (layerId: string): void => {
  layerPropertiesRequestStore.setSnapshot({ request: { layerId, token: nextToken++ } });
};

export const clearLayerPropertiesRequest = (token?: number): void => {
  const current = layerPropertiesRequestStore.getSnapshot().request;
  if (token !== undefined && current?.token !== token) {
    return;
  }
  layerPropertiesRequestStore.setSnapshot({ request: null });
};

export const getLayerPropertiesRequest = (): LayerPropertiesRequest | null =>
  layerPropertiesRequestStore.getSnapshot().request;

export const useCurrentLayerPropertiesRequest = (): LayerPropertiesRequest | null =>
  layerPropertiesRequestStore.useSelector((snapshot) => snapshot.request, Object.is);
