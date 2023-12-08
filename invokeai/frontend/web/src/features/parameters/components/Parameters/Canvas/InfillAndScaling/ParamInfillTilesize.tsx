import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setInfillTileSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector([stateSelector], ({ generation }) => {
  const { infillTileSize, infillMethod } = generation;

  return {
    infillTileSize,
    infillMethod,
  };
});

const ParamInfillTileSize = () => {
  const dispatch = useAppDispatch();
  const { infillTileSize, infillMethod } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setInfillTileSize(v));
    },
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(setInfillTileSize(32));
  }, [dispatch]);

  return (
    <IAISlider
      isDisabled={infillMethod !== 'tile'}
      label={t('parameters.tileSize')}
      min={16}
      max={64}
      sliderNumberInputProps={{ max: 256 }}
      value={infillTileSize}
      onChange={handleChange}
      withInput
      withSliderMarks
      withReset
      handleReset={handleReset}
    />
  );
};

export default memo(ParamInfillTileSize);
