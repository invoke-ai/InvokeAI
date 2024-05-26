import type { ProcessorConfig } from 'features/controlLayers/util/controlAdapters';

export type ProcessorComponentProps<T extends ProcessorConfig> = {
  onChange: (config: T) => void;
  config: T;
};
