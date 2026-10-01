import { useAppSelector } from 'app/store/storeHooks';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import { selectSystemShouldShowTokenCounter } from 'features/system/store/systemSlice';
import { useMemo } from 'react';
import { useDebounce } from 'use-debounce';

import { calculatePromptTokens } from './tokenizers';
import type { TokenCountResult } from './types';

export const usePromptTokenCount = (promptText: string): TokenCountResult | null => {
  const isEnabled = useAppSelector(selectSystemShouldShowTokenCounter);
  const model = useAppSelector(selectModel);
  const baseModel = model?.base;

  const [debouncedText] = useDebounce(promptText, 300);

  const result = useMemo(() => {
    // Performance rule: Completely skip all work when the toggle is off
    if (!isEnabled) {
      return null;
    }

    return calculatePromptTokens(debouncedText, baseModel);
  }, [isEnabled, debouncedText, baseModel]);

  return result;
};
