import type { FilterConfig } from 'features/controlLayers/store/filters';

export type FilterComponentProps<T extends FilterConfig> = {
  onChange: (config: T) => void;
  config: T;
};
