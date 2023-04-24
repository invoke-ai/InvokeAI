import { Box, BoxProps } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setWidth } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const WidthSlider = (props: BoxProps) => {
  const width = useAppSelector((state: RootState) => state.generation.width);
  const shift = useAppSelector((state: RootState) => state.hotkeys.shift);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  return (
    <Box {...props}>
      <IAISlider
        isDisabled={activeTabName === 'unifiedCanvas'}
        label={t('parameters.width')}
        value={width}
        min={64}
        step={shift ? 8 : 64}
        max={2048}
        onChange={(v) => dispatch(setWidth(v))}
        handleReset={() => dispatch(setWidth(512))}
        withInput
        withReset
        withSliderMarks
        sliderNumberInputProps={{ max: 15360 }}
      />
    </Box>
  );
};

export default memo(WidthSlider);
