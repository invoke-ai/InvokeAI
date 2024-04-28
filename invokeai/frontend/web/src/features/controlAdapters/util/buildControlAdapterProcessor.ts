import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { isControlAdapterProcessorType } from 'features/controlAdapters/store/types';
import type { ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';

export const buildControlAdapterProcessor = (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig) => {
  const defaultPreprocessor = modelConfig.default_settings?.preprocessor;
  const processorType = isControlAdapterProcessorType(defaultPreprocessor) ? defaultPreprocessor : 'none';
  const processorNode = CONTROLNET_PROCESSORS[processorType].buildDefaults(modelConfig.base);

  return { processorType, processorNode };
};
