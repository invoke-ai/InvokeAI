import { createAction } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { controlAdapterAdded, controlAdapterRemoved } from 'features/controlAdapters/store/controlAdaptersSlice';
import {
  controlAdapterLayerAdded,
  ipAdapterLayerAdded,
  layerDeleted,
  maskedGuidanceLayerAdded,
  maskLayerIPAdapterAdded,
  maskLayerIPAdapterDeleted,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import type { Layer } from 'features/regionalPrompts/store/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export const guidanceLayerAdded = createAction<Layer['type']>('regionalPrompts/guidanceLayerAdded');
export const guidanceLayerDeleted = createAction<string>('regionalPrompts/guidanceLayerDeleted');
export const allLayersDeleted = createAction('regionalPrompts/allLayersDeleted');
export const guidanceLayerIPAdapterAdded = createAction<string>('regionalPrompts/guidanceLayerIPAdapterAdded');
export const guidanceLayerIPAdapterDeleted = createAction<{ layerId: string; ipAdapterId: string }>(
  'regionalPrompts/guidanceLayerIPAdapterDeleted'
);

export const addRegionalControlToControlAdapterBridge = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: guidanceLayerAdded,
    effect: (action, { dispatch }) => {
      const type = action.payload;
      const layerId = uuidv4();
      if (type === 'ip_adapter_layer') {
        const ipAdapterId = uuidv4();
        dispatch(controlAdapterAdded({ type: 'ip_adapter', overrides: { id: ipAdapterId } }));
        dispatch(ipAdapterLayerAdded({ layerId, ipAdapterId }));
      } else if (type === 'control_adapter_layer') {
        const controlNetId = uuidv4();
        dispatch(controlAdapterAdded({ type: 'controlnet', overrides: { id: controlNetId } }));
        dispatch(controlAdapterLayerAdded({ layerId, controlNetId }));
      } else if (type === 'masked_guidance_layer') {
        dispatch(maskedGuidanceLayerAdded({ layerId }));
      }
    },
  });

  startAppListening({
    actionCreator: guidanceLayerDeleted,
    effect: (action, { getState, dispatch }) => {
      const layerId = action.payload;
      const state = getState();
      const layer = state.regionalPrompts.present.layers.find((l) => l.id === layerId);
      assert(layer, `Layer ${layerId} not found`);

      if (layer.type === 'ip_adapter_layer') {
        dispatch(controlAdapterRemoved({ id: layer.ipAdapterId }));
      } else if (layer.type === 'control_adapter_layer') {
        dispatch(controlAdapterRemoved({ id: layer.controlNetId }));
      } else if (layer.type === 'masked_guidance_layer') {
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
      for (const layer of state.regionalPrompts.present.layers) {
        dispatch(guidanceLayerDeleted(layer.id));
      }
    },
  });

  startAppListening({
    actionCreator: guidanceLayerIPAdapterAdded,
    effect: (action, { dispatch }) => {
      const layerId = action.payload;
      const ipAdapterId = uuidv4();
      dispatch(controlAdapterAdded({ type: 'ip_adapter', overrides: { id: ipAdapterId } }));
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
