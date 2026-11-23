import { useActiveProjectSelector } from '@workbench/WorkbenchContext';

import { getSelectedCfgScale, getSelectedModelBase } from './selectedModel';

/** Share the Generate model base across control/regional settings and layer creation so adapter kinds agree. */
export const useSelectedModelBase = (): string | null => {
  return useActiveProjectSelector(getSelectedModelBase);
};

/** The Generate CFG Scale, which decides whether some bases read regional negatives; null while unset. */
export const useSelectedCfgScale = (): number | null => {
  return useActiveProjectSelector(getSelectedCfgScale);
};
