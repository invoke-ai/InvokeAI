import { Box, BoxProps } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setHeight } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

import { useTranslation } from 'react-i18next';

const HeightSlider = (props: BoxProps) => {
  const height = useAppSelector((state: RootState) => state.generation.height);
  const shift = useAppSelector((state: RootState) => state.hotkeys.shift);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  return (
    <Box {...props}>
      <IAISlider
        isDisabled={activeTabName === 'unifiedCanvas'}
        label={t('parameters.height')}
        value={height}
        min={64}
        step={shift ? 8 : 64}
        max={2048}
        onChange={(v) => dispatch(setHeight(v))}
        handleReset={() => dispatch(setHeight(512))}
        withInput
        withReset
        withSliderMarks
        sliderNumberInputProps={{ max: 15360 }}
      />
    </Box>
  );
};

export default memo(HeightSlider);
