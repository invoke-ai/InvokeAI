import { useActiveProjectSelector } from '@workbench/WorkbenchContext';

import { getSelectedModelBase } from './selectedModel';

/** Share the Generate model base across control/regional settings and layer creation so adapter kinds agree. */
export const useSelectedModelBase = (): string | null => {
  return useActiveProjectSelector(getSelectedModelBase);
};
