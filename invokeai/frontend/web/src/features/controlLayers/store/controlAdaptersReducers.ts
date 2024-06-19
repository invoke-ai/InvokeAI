import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { IRect } from 'konva/lib/types';
import { isEqual } from 'lodash-es';
import type { ControlNetModelConfig, ImageDTO, T2IAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import type {
  CanvasV2State,
  ControlAdapterEntity,
  ControlModeV2,
  ControlNetConfig,
  ControlNetData,
  Filter,
  ProcessorConfig,
  T2IAdapterConfig,
  T2IAdapterData,
} from './types';
import { buildControlAdapterProcessorV2, imageDTOToImageObject } from './types';

export const selectCA = (state: CanvasV2State, id: string) => state.controlAdapters.find((ca) => ca.id === id);
export const selectCAOrThrow = (state: CanvasV2State, id: string) => {
  const ca = selectCA(state, id);
  assert(ca, `Control Adapter with id ${id} not found`);
  return ca;
};

export const controlAdaptersReducers = {
  caAdded: {
    reducer: (state, action: PayloadAction<{ id: string; config: ControlNetConfig | T2IAdapterConfig }>) => {
      const { id, config } = action.payload;
      state.controlAdapters.push({
        id,
        type: 'control_adapter',
        x: 0,
        y: 0,
        bbox: null,
        bboxNeedsUpdate: false,
        isEnabled: true,
        opacity: 1,
        filter: 'LightnessToAlphaFilter',
        processorPendingBatchId: null,
        ...config,
      });
      state.selectedEntityIdentifier = { type: 'control_adapter', id };
    },
    prepare: (payload: { config: ControlNetConfig | T2IAdapterConfig }) => ({
      payload: { id: uuidv4(), ...payload },
    }),
  },
  caRecalled: (state, action: PayloadAction<{ data: ControlAdapterEntity }>) => {
    const { data } = action.payload;
    state.controlAdapters.push(data);
    state.selectedEntityIdentifier = { type: 'control_adapter', id: data.id };
  },
  caIsEnabledToggled: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    ca.isEnabled = !ca.isEnabled;
  },
  caTranslated: (state, action: PayloadAction<{ id: string; x: number; y: number }>) => {
    const { id, x, y } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    ca.x = x;
    ca.y = y;
  },
  caBboxChanged: (state, action: PayloadAction<{ id: string; bbox: IRect | null }>) => {
    const { id, bbox } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    ca.bbox = bbox;
    ca.bboxNeedsUpdate = false;
  },
  caDeleted: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    state.controlAdapters = state.controlAdapters.filter((ca) => ca.id !== id);
  },
  caAllDeleted: (state) => {
    state.controlAdapters = [];
  },
  caOpacityChanged: (state, action: PayloadAction<{ id: string; opacity: number }>) => {
    const { id, opacity } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    ca.opacity = opacity;
  },
  caMovedForwardOne: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    moveOneToEnd(state.controlAdapters, ca);
  },
  caMovedToFront: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    moveToEnd(state.controlAdapters, ca);
  },
  caMovedBackwardOne: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    moveOneToStart(state.controlAdapters, ca);
  },
  caMovedToBack: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    moveToStart(state.controlAdapters, ca);
  },
  caImageChanged: {
    reducer: (state, action: PayloadAction<{ id: string; imageDTO: ImageDTO | null; objectId: string }>) => {
      const { id, imageDTO, objectId } = action.payload;
      const ca = selectCA(state, id);
      if (!ca) {
        return;
      }
      ca.bbox = null;
      ca.bboxNeedsUpdate = true;
      ca.isEnabled = true;
      if (imageDTO) {
        const newImageObject = imageDTOToImageObject(id, objectId, imageDTO);
        if (isEqual(newImageObject, ca.imageObject)) {
          return;
        }
        ca.imageObject = newImageObject;
        ca.processedImageObject = null;
      } else {
        ca.imageObject = null;
        ca.processedImageObject = null;
      }
    },
    prepare: (payload: { id: string; imageDTO: ImageDTO | null }) => ({ payload: { ...payload, objectId: uuidv4() } }),
  },
  caProcessedImageChanged: {
    reducer: (state, action: PayloadAction<{ id: string; imageDTO: ImageDTO | null; objectId: string }>) => {
      const { id, imageDTO, objectId } = action.payload;
      const ca = selectCA(state, id);
      if (!ca) {
        return;
      }
      ca.bbox = null;
      ca.bboxNeedsUpdate = true;
      ca.isEnabled = true;
      ca.processedImageObject = imageDTO ? imageDTOToImageObject(id, objectId, imageDTO) : null;
    },
    prepare: (payload: { id: string; imageDTO: ImageDTO | null }) => ({ payload: { ...payload, objectId: uuidv4() } }),
  },
  caModelChanged: (
    state,
    action: PayloadAction<{
      id: string;
      modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | null;
    }>
  ) => {
    const { id, modelConfig } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    if (!modelConfig) {
      ca.model = null;
      return;
    }
    ca.model = zModelIdentifierField.parse(modelConfig);

    const candidateProcessorConfig = buildControlAdapterProcessorV2(modelConfig);
    if (candidateProcessorConfig?.type !== ca.processorConfig?.type) {
      // The processor has changed. For example, the previous model was a Canny model and the new model is a Depth
      // model. We need to use the new processor.
      ca.processedImageObject = null;
      ca.processorConfig = candidateProcessorConfig;
    }

    // We may need to convert the CA to match the model
    if (ca.adapterType === 't2i_adapter' && ca.model.type === 'controlnet') {
      const convertedCA: ControlNetData = { ...ca, adapterType: 'controlnet', controlMode: 'balanced' };
      state.controlAdapters.splice(state.controlAdapters.indexOf(ca), 1, convertedCA);
    } else if (ca.adapterType === 'controlnet' && ca.model.type === 't2i_adapter') {
      const { controlMode: _, ...rest } = ca;
      const convertedCA: T2IAdapterData = { ...rest, adapterType: 't2i_adapter' };
      state.controlAdapters.splice(state.controlAdapters.indexOf(ca), 1, convertedCA);
    }
  },
  caControlModeChanged: (state, action: PayloadAction<{ id: string; controlMode: ControlModeV2 }>) => {
    const { id, controlMode } = action.payload;
    const ca = selectCA(state, id);
    if (!ca || ca.adapterType !== 'controlnet') {
      return;
    }
    ca.controlMode = controlMode;
  },
  caProcessorConfigChanged: (state, action: PayloadAction<{ id: string; processorConfig: ProcessorConfig | null }>) => {
    const { id, processorConfig } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    ca.processorConfig = processorConfig;
    if (!processorConfig) {
      ca.processedImageObject = null;
    }
  },
  caFilterChanged: (state, action: PayloadAction<{ id: string; filter: Filter }>) => {
    const { id, filter } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    ca.filter = filter;
  },
  caProcessorPendingBatchIdChanged: (state, action: PayloadAction<{ id: string; batchId: string | null }>) => {
    const { id, batchId } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    ca.processorPendingBatchId = batchId;
  },
  caWeightChanged: (state, action: PayloadAction<{ id: string; weight: number }>) => {
    const { id, weight } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    ca.weight = weight;
  },
  caBeginEndStepPctChanged: (state, action: PayloadAction<{ id: string; beginEndStepPct: [number, number] }>) => {
    const { id, beginEndStepPct } = action.payload;
    const ca = selectCA(state, id);
    if (!ca) {
      return;
    }
    ca.beginEndStepPct = beginEndStepPct;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
