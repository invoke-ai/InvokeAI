import { HEIGHTS } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import IAISlider from 'common/components/IAISlider';
import { setHeight } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';

import { useTranslation } from 'react-i18next';

export default function MainHeight() {
  const height = useAppSelector((state: RootState) => state.generation.height);
  const shouldUseSliders = useAppSelector(
    (state: RootState) => state.ui.shouldUseSliders
  );
  const activeTabName = useAppSelector(activeTabNameSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  return shouldUseSliders ? (
    <IAISlider
      isSliderDisabled={activeTabName === 'unifiedCanvas'}
      isInputDisabled={activeTabName === 'unifiedCanvas'}
      isResetDisabled={activeTabName === 'unifiedCanvas'}
      label={t('parameters.height')}
      value={height}
      min={64}
      step={64}
      max={2048}
      onChange={(v) => dispatch(setHeight(v))}
      handleReset={() => dispatch(setHeight(512))}
      withInput
      withReset
      withSliderMarks
      sliderNumberInputProps={{ max: 15360 }}
    />
  ) : (
    <IAISelect
      isDisabled={activeTabName === 'unifiedCanvas'}
      label={t('parameters.height')}
      value={height}
      flexGrow={1}
      onChange={(e) => dispatch(setHeight(Number(e.target.value)))}
      validValues={HEIGHTS}
    />
  );
}
