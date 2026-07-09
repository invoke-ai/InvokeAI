import { createExternalStore } from '@workbench/externalStore';

export type LayerPropertiesSection = 'filter';

export interface LayerPropertiesRequest {
  layerId: string;
  section: LayerPropertiesSection;
  token: number;
}

export const layerPropertiesRequestStore = createExternalStore<{ request: LayerPropertiesRequest | null }>({
  request: null,
});

let nextToken = 1;

export const requestLayerProperties = (layerId: string, section: LayerPropertiesSection): void => {
  layerPropertiesRequestStore.setSnapshot({ request: { layerId, section, token: nextToken++ } });
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

export const useLayerPropertiesRequest = (layerId: string): LayerPropertiesRequest | null =>
  layerPropertiesRequestStore.useSelector(
    (snapshot) => (snapshot.request?.layerId === layerId ? snapshot.request : null),
    Object.is
  );
