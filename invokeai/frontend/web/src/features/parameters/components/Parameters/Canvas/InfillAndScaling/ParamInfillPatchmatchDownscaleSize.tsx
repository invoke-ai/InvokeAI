import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setInfillPatchmatchDownscaleSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';

import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [generationSelector],
  (parameters) => {
    const { infillPatchmatchDownscaleSize, infillMethod } = parameters;

    return {
      infillPatchmatchDownscaleSize,
      infillMethod,
    };
  },
  defaultSelectorOptions
);

const ParamInfillPatchmatchDownscaleSize = () => {
  const dispatch = useAppDispatch();
  const { infillPatchmatchDownscaleSize, infillMethod } =
    useAppSelector(selector);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setInfillPatchmatchDownscaleSize(v));
    },
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(setInfillPatchmatchDownscaleSize(2));
  }, [dispatch]);

  return (
    <IAISlider
      isDisabled={infillMethod !== 'patchmatch'}
      label={t('parameters.patchmatchDownScaleSize')}
      min={1}
      max={10}
      value={infillPatchmatchDownscaleSize}
      onChange={handleChange}
      withInput
      withSliderMarks
      withReset
      handleReset={handleReset}
    />
  );
};

export default memo(ParamInfillPatchmatchDownscaleSize);
