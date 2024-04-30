import { deepClone } from 'common/util/deepClone';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type {
  ControlAdapterConfig,
  ControlAdapterType,
  ControlNetConfig,
  IPAdapterConfig,
  RequiredCannyImageProcessorInvocation,
  T2IAdapterConfig,
} from 'features/controlAdapters/store/types';
import { merge } from 'lodash-es';

export const initialControlNet: Omit<ControlNetConfig, 'id'> = {
  type: 'controlnet',
  isEnabled: true,
  model: null,
  weight: 1,
  beginStepPct: 0,
  endStepPct: 1,
  controlMode: 'balanced',
  resizeMode: 'just_resize',
  controlImage: null,
  controlImageDimensions: null,
  processedControlImage: null,
  processedControlImageDimensions: null,
  processorType: 'canny_image_processor',
  processorNode: CONTROLNET_PROCESSORS.canny_image_processor.buildDefaults() as RequiredCannyImageProcessorInvocation,
  shouldAutoConfig: true,
};

export const initialT2IAdapter: Omit<T2IAdapterConfig, 'id'> = {
  type: 't2i_adapter',
  isEnabled: true,
  model: null,
  weight: 1,
  beginStepPct: 0,
  endStepPct: 1,
  resizeMode: 'just_resize',
  controlImage: null,
  controlImageDimensions: null,
  processedControlImage: null,
  processedControlImageDimensions: null,
  processorType: 'canny_image_processor',
  processorNode: CONTROLNET_PROCESSORS.canny_image_processor.buildDefaults() as RequiredCannyImageProcessorInvocation,
  shouldAutoConfig: true,
};

export const initialIPAdapter: Omit<IPAdapterConfig, 'id'> = {
  type: 'ip_adapter',
  isEnabled: true,
  controlImage: null,
  model: null,
  method: 'full',
  clipVisionModel: 'ViT-H',
  weight: 1,
  beginStepPct: 0,
  endStepPct: 1,
};

export const buildControlAdapter = (
  id: string,
  type: ControlAdapterType,
  overrides: Partial<ControlAdapterConfig> = {}
): ControlAdapterConfig => {
  switch (type) {
    case 'controlnet':
      return merge(deepClone(initialControlNet), { id, ...overrides });
    case 't2i_adapter':
      return merge(deepClone(initialT2IAdapter), { id, ...overrides });
    case 'ip_adapter':
      return merge(deepClone(initialIPAdapter), { id, ...overrides });
    default:
      throw new Error(`Unknown control adapter type: ${type}`);
  }
};
