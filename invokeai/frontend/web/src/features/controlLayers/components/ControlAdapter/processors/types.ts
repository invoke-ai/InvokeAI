import type { ProcessorConfig } from 'features/controlLayers/store/types';

export type ProcessorComponentProps<T extends ProcessorConfig> = {
  onChange: (config: T) => void;
  config: T;
};
