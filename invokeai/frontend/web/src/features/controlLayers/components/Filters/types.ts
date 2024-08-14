import type { FilterConfig } from 'features/controlLayers/store/types';

export type FilterComponentProps<T extends FilterConfig> = {
  onChange: (config: T) => void;
  config: T;
};
