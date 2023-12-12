import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setInfillPatchmatchDownscaleSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector([stateSelector], ({ generation }) => {
  const { infillPatchmatchDownscaleSize, infillMethod } = generation;

  return {
    infillPatchmatchDownscaleSize,
    infillMethod,
  };
});

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
