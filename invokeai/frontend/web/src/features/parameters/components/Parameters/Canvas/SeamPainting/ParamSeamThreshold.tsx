import {
  FormControl,
  FormLabel,
  HStack,
  RangeSlider,
  RangeSliderFilledTrack,
  RangeSliderMark,
  RangeSliderThumb,
  RangeSliderTrack,
  Tooltip,
} from '@chakra-ui/react';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  setSeamHighThreshold,
  setSeamLowThreshold,
} from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { BiReset } from 'react-icons/bi';

const ParamSeamThreshold = () => {
  const dispatch = useAppDispatch();
  const seamLowThreshold = useAppSelector(
    (state: RootState) => state.generation.seamLowThreshold
  );

  const seamHighThreshold = useAppSelector(
    (state: RootState) => state.generation.seamHighThreshold
  );
  const { t } = useTranslation();

  const handleSeamThresholdChange = useCallback(
    (v: number[]) => {
      dispatch(setSeamLowThreshold(v[0] as number));
      dispatch(setSeamHighThreshold(v[1] as number));
    },
    [dispatch]
  );

  const handleSeamThresholdReset = () => {
    dispatch(setSeamLowThreshold(100));
    dispatch(setSeamHighThreshold(200));
  };

  return (
    <FormControl>
      <FormLabel>{t('parameters.seamThreshold')}</FormLabel>
      <HStack w="100%" gap={4} mt={-2}>
        <RangeSlider
          aria-label={[
            t('parameters.seamLowThreshold'),
            t('parameters.seamHighThreshold'),
          ]}
          value={[seamLowThreshold, seamHighThreshold]}
          min={0}
          max={255}
          step={1}
          minStepsBetweenThumbs={1}
          onChange={handleSeamThresholdChange}
        >
          <RangeSliderTrack>
            <RangeSliderFilledTrack />
          </RangeSliderTrack>
          <Tooltip label={seamLowThreshold} placement="top" hasArrow>
            <RangeSliderThumb index={0} />
          </Tooltip>
          <Tooltip label={seamHighThreshold} placement="top" hasArrow>
            <RangeSliderThumb index={1} />
          </Tooltip>
          <RangeSliderMark
            value={0}
            sx={{
              insetInlineStart: '0 !important',
              insetInlineEnd: 'unset !important',
            }}
          >
            0
          </RangeSliderMark>
          <RangeSliderMark
            value={0.392}
            sx={{
              insetInlineStart: '38.4% !important',
              transform: 'translateX(-38.4%)',
            }}
          >
            100
          </RangeSliderMark>
          <RangeSliderMark
            value={0.784}
            sx={{
              insetInlineStart: '79.8% !important',
              transform: 'translateX(-79.8%)',
            }}
          >
            200
          </RangeSliderMark>
          <RangeSliderMark
            value={1}
            sx={{
              insetInlineStart: 'unset !important',
              insetInlineEnd: '0 !important',
            }}
          >
            255
          </RangeSliderMark>
        </RangeSlider>
        <IAIIconButton
          size="sm"
          aria-label={t('accessibility.reset')}
          tooltip={t('accessibility.reset')}
          icon={<BiReset />}
          onClick={handleSeamThresholdReset}
        />
      </HStack>
    </FormControl>
  );
};

export default memo(ParamSeamThreshold);
