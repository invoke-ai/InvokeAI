import type { TabName } from 'features/ui/store/uiTypes';
import { type AnyModelConfigWithExternal, isExternalApiModelConfig } from 'services/api/types';

export const isExternalModelUnsupportedForTab = (model: AnyModelConfigWithExternal, tab: TabName): boolean => {
  if (!isExternalApiModelConfig(model)) {
    return false;
  }

  if (tab === 'generate') {
    return !model.capabilities.modes.includes('txt2img');
  }

  return false;
};
