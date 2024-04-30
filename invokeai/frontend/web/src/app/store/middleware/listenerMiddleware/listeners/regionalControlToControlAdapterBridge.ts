import { createAction } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { controlAdapterAdded, controlAdapterRemoved } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlNetConfig, IPAdapterConfig } from 'features/controlAdapters/store/types';
import { isControlAdapterProcessorType } from 'features/controlAdapters/store/types';
import {
  controlAdapterLayerAdded,
  ipAdapterLayerAdded,
  layerDeleted,
  maskedGuidanceLayerAdded,
  maskLayerIPAdapterAdded,
  maskLayerIPAdapterDeleted,
} from 'features/controlLayers/store/controlLayersSlice';
import type { Layer } from 'features/controlLayers/store/types';
import { modelConfigsAdapterSelectors, modelsApi } from 'services/api/endpoints/models';
import { isControlNetModelConfig, isIPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export const guidanceLayerAdded = createAction<Layer['type']>('controlLayers/guidanceLayerAdded');
export const guidanceLayerDeleted = createAction<string>('controlLayers/guidanceLayerDeleted');
export const allLayersDeleted = createAction('controlLayers/allLayersDeleted');
export const guidanceLayerIPAdapterAdded = createAction<string>('controlLayers/guidanceLayerIPAdapterAdded');
export const guidanceLayerIPAdapterDeleted = createAction<{ layerId: string; ipAdapterId: string }>(
  'controlLayers/guidanceLayerIPAdapterDeleted'
);

export const addRegionalControlToControlAdapterBridge = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: guidanceLayerAdded,
    effect: (action, { dispatch, getState }) => {
      const type = action.payload;
      const layerId = uuidv4();
      if (type === 'regional_guidance_layer') {
        dispatch(maskedGuidanceLayerAdded({ layerId }));
        return;
      }

      const state = getState();
      const baseModel = state.generation.model?.base;
      const modelConfigs = modelsApi.endpoints.getModelConfigs.select(undefined)(state).data;

      if (type === 'ip_adapter_layer') {
        const ipAdapterId = uuidv4();
        const overrides: Partial<IPAdapterConfig> = {
          id: ipAdapterId,
        };

        // Find and select the first matching model
        if (modelConfigs) {
          const models = modelConfigsAdapterSelectors.selectAll(modelConfigs).filter(isIPAdapterModelConfig);
          overrides.model = models.find((m) => m.base === baseModel) ?? null;
        }
        dispatch(controlAdapterAdded({ type: 'ip_adapter', overrides }));
        dispatch(ipAdapterLayerAdded({ layerId, ipAdapterId }));
        return;
      }

      if (type === 'control_adapter_layer') {
        const controlNetId = uuidv4();
        const overrides: Partial<ControlNetConfig> = {
          id: controlNetId,
        };

        // Find and select the first matching model
        if (modelConfigs) {
          const models = modelConfigsAdapterSelectors.selectAll(modelConfigs).filter(isControlNetModelConfig);
          const model = models.find((m) => m.base === baseModel) ?? null;
          overrides.model = model;
          const defaultPreprocessor = model?.default_settings?.preprocessor;
          overrides.processorType = isControlAdapterProcessorType(defaultPreprocessor) ? defaultPreprocessor : 'none';
          overrides.processorNode = CONTROLNET_PROCESSORS[overrides.processorType].buildDefaults(baseModel);
        }
        dispatch(controlAdapterAdded({ type: 'controlnet', overrides }));
        dispatch(controlAdapterLayerAdded({ layerId, controlNetId }));
        return;
      }
    },
  });

  startAppListening({
    actionCreator: guidanceLayerDeleted,
    effect: (action, { getState, dispatch }) => {
      const layerId = action.payload;
      const state = getState();
      const layer = state.controlLayers.present.layers.find((l) => l.id === layerId);
      assert(layer, `Layer ${layerId} not found`);

      if (layer.type === 'ip_adapter_layer') {
        dispatch(controlAdapterRemoved({ id: layer.ipAdapterId }));
      } else if (layer.type === 'control_adapter_layer') {
        dispatch(controlAdapterRemoved({ id: layer.controlNetId }));
      } else if (layer.type === 'regional_guidance_layer') {
        for (const ipAdapterId of layer.ipAdapterIds) {
          dispatch(controlAdapterRemoved({ id: ipAdapterId }));
        }
      }
      dispatch(layerDeleted(layerId));
    },
  });

  startAppListening({
    actionCreator: allLayersDeleted,
    effect: (action, { dispatch, getOriginalState }) => {
      const state = getOriginalState();
      for (const layer of state.controlLayers.present.layers) {
        dispatch(guidanceLayerDeleted(layer.id));
      }
    },
  });

  startAppListening({
    actionCreator: guidanceLayerIPAdapterAdded,
    effect: (action, { dispatch, getState }) => {
      const layerId = action.payload;
      const ipAdapterId = uuidv4();
      const overrides: Partial<IPAdapterConfig> = {
        id: ipAdapterId,
      };

      // Find and select the first matching model
      const state = getState();
      const baseModel = state.generation.model?.base;
      const modelConfigs = modelsApi.endpoints.getModelConfigs.select(undefined)(state).data;
      if (modelConfigs) {
        const models = modelConfigsAdapterSelectors.selectAll(modelConfigs).filter(isIPAdapterModelConfig);
        overrides.model = models.find((m) => m.base === baseModel) ?? null;
      }

      dispatch(controlAdapterAdded({ type: 'ip_adapter', overrides }));
      dispatch(maskLayerIPAdapterAdded({ layerId, ipAdapterId }));
    },
  });

  startAppListening({
    actionCreator: guidanceLayerIPAdapterDeleted,
    effect: (action, { dispatch }) => {
      const { layerId, ipAdapterId } = action.payload;
      dispatch(controlAdapterRemoved({ id: ipAdapterId }));
      dispatch(maskLayerIPAdapterDeleted({ layerId, ipAdapterId }));
    },
  });
};
