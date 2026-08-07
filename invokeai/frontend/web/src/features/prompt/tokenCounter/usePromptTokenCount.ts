import { useAppSelector } from 'app/store/storeHooks';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import { selectSystemShouldShowTokenCounter } from 'features/system/store/systemSlice';
import { useEffect, useState } from 'react';
import { useDebounce } from 'use-debounce';

import { calculatePromptTokens, getTokenizerConfig } from './tokenizers';
import type { TokenCountResult } from './types';

export const usePromptTokenCount = (promptText: string): TokenCountResult | null => {
  const isEnabled = useAppSelector(selectSystemShouldShowTokenCounter);
  const model = useAppSelector(selectModel);
  const baseModel = model?.base;

  const [debouncedText] = useDebounce(promptText, 300);
  const [result, setResult] = useState<TokenCountResult | null>(null);

  useEffect(() => {
    // Performance rule: Completely skip all work when the toggle is off
    if (!isEnabled) {
      setResult(null);
      return;
    }

    let isMounted = true;

    void calculatePromptTokens(debouncedText, baseModel).then((res) => {
      if (isMounted) {
        setResult(res);
      }
    });

    return () => {
      isMounted = false;
    };
  }, [isEnabled, debouncedText, baseModel]);

  if (!isEnabled) {
    return null;
  }

  if (!result) {
    const { family, limit } = getTokenizerConfig(baseModel);
    return {
      count: 0,
      limit,
      tokenizerFamily: family,
      isNearLimit: false,
      isOverLimit: false,
    };
  }

  return result;
};
