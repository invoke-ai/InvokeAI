import { useActiveProjectSelector } from '@workbench/WorkbenchContext';

import { getSelectedModelBase } from './selectedModel';

/**
 * The main model's base (`sd-1` / `sdxl` / `flux` / …), read from the generate
 * widget values. Drives which control-adapter kinds and reference-image kinds a
 * region can consume. Shared by the control/regional settings and the add-layer
 * flow so a freshly created "regional reference image" mints the base-correct kind.
 */
export const useSelectedModelBase = (): string | null => {
  return useActiveProjectSelector(getSelectedModelBase);
};
