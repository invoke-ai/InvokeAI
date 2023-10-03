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
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  ipAdapterBeginStepPctChanged,
  ipAdapterEndStepPctChanged,
} from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const formatPct = (v: number) => `${Math.round(v * 100)}%`;

const ParamIPAdapterBeginEnd = () => {
  const isEnabled = useAppSelector(
    (state: RootState) => state.controlNet.isIPAdapterEnabled
  );
  const beginStepPct = useAppSelector(
    (state: RootState) => state.controlNet.ipAdapterInfo.beginStepPct
  );
  const endStepPct = useAppSelector(
    (state: RootState) => state.controlNet.ipAdapterInfo.endStepPct
  );
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleStepPctChanged = useCallback(
    (v: number[]) => {
      dispatch(ipAdapterBeginStepPctChanged(v[0] as number));
      dispatch(ipAdapterEndStepPctChanged(v[1] as number));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={!isEnabled}>
      <FormLabel>{t('controlnet.beginEndStepPercent')}</FormLabel>
      <HStack w="100%" gap={2} alignItems="center">
        <RangeSlider
          aria-label={['Begin Step %', 'End Step %!']}
          value={[beginStepPct, endStepPct]}
          onChange={handleStepPctChanged}
          min={0}
          max={1}
          step={0.01}
          minStepsBetweenThumbs={5}
          isDisabled={!isEnabled}
        >
          <RangeSliderTrack>
            <RangeSliderFilledTrack />
          </RangeSliderTrack>
          <Tooltip label={formatPct(beginStepPct)} placement="top" hasArrow>
            <RangeSliderThumb index={0} />
          </Tooltip>
          <Tooltip label={formatPct(endStepPct)} placement="top" hasArrow>
            <RangeSliderThumb index={1} />
          </Tooltip>
          <RangeSliderMark
            value={0}
            sx={{
              insetInlineStart: '0 !important',
              insetInlineEnd: 'unset !important',
            }}
          >
            0%
          </RangeSliderMark>
          <RangeSliderMark
            value={0.5}
            sx={{
              insetInlineStart: '50% !important',
              transform: 'translateX(-50%)',
            }}
          >
            50%
          </RangeSliderMark>
          <RangeSliderMark
            value={1}
            sx={{
              insetInlineStart: 'unset !important',
              insetInlineEnd: '0 !important',
            }}
          >
            100%
          </RangeSliderMark>
        </RangeSlider>
      </HStack>
    </FormControl>
  );
};

export default memo(ParamIPAdapterBeginEnd);
