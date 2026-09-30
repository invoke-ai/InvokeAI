import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  seedreamOptimizePromptChanged,
  seedreamWatermarkChanged,
  selectSeedreamOptimizePrompt,
  selectSeedreamWatermark,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const SeedreamProviderOptions = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const watermark = useAppSelector(selectSeedreamWatermark);
  const optimizePrompt = useAppSelector(selectSeedreamOptimizePrompt);

  const onWatermarkChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(seedreamWatermarkChanged(e.target.checked)),
    [dispatch]
  );

  const onOptimizePromptChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(seedreamOptimizePromptChanged(e.target.checked)),
    [dispatch]
  );

  return (
    <>
      <FormControl>
        <FormLabel>{t('parameters.watermark', 'Watermark')}</FormLabel>
        <Checkbox isChecked={watermark} onChange={onWatermarkChange} />
      </FormControl>
      <FormControl>
        <FormLabel>{t('parameters.optimizePrompt', 'Optimize Prompt')}</FormLabel>
        <Checkbox isChecked={optimizePrompt} onChange={onOptimizePromptChange} />
      </FormControl>
    </>
  );
});

SeedreamProviderOptions.displayName = 'SeedreamProviderOptions';
