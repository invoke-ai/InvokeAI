import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamSize } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function SeamSize() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const seamSize = useAppSelector(
    (state: RootState) => state.generation.seamSize
  );

  return (
    <IAISlider
      sliderMarkRightOffset={-6}
      label={t('parameters.seamSize')}
      min={1}
      max={256}
      sliderNumberInputProps={{ max: 512 }}
      value={seamSize}
      onChange={(v) => {
        dispatch(setSeamSize(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => dispatch(setSeamSize(96))}
    />
  );
}
