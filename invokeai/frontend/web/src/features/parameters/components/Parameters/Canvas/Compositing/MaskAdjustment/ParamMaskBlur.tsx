import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { CompositingBlurPopover } from 'features/informationalPopovers/components/compositingBlur';
import { setMaskBlur } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function ParamMaskBlur() {
  const dispatch = useAppDispatch();
  const maskBlur = useAppSelector(
    (state: RootState) => state.generation.maskBlur
  );
  const { t } = useTranslation();

  return (
    <CompositingBlurPopover>
      <IAISlider
        label={t('parameters.maskBlur')}
        min={0}
        max={64}
        sliderNumberInputProps={{ max: 512 }}
        value={maskBlur}
        onChange={(v) => {
          dispatch(setMaskBlur(v));
        }}
        withInput
        withSliderMarks
        withReset
        handleReset={() => {
          dispatch(setMaskBlur(16));
        }}
      />
    </CompositingBlurPopover>
  );
}
