import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setTileSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';

import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [generationSelector],
  (parameters) => {
    const { tileSize, infillMethod } = parameters;

    return {
      tileSize,
      infillMethod,
    };
  },
  defaultSelectorOptions
);

const ParamInfillTileSize = () => {
  const dispatch = useAppDispatch();
  const { tileSize, infillMethod } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setTileSize(v));
    },
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(setTileSize(32));
  }, [dispatch]);

  return (
    <IAISlider
      isDisabled={infillMethod !== 'tile'}
      label={t('parameters.tileSize')}
      min={16}
      max={64}
      sliderNumberInputProps={{ max: 256 }}
      value={tileSize}
      onChange={handleChange}
      withInput
      withSliderMarks
      withReset
      handleReset={handleReset}
    />
  );
};

export default memo(ParamInfillTileSize);
