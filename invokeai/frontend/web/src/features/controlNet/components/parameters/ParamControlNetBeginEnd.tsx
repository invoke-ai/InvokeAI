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
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
} from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { BiReset } from 'react-icons/bi';

type Props = {
  controlNetId: string;
  beginStepPct: number;
  endStepPct: number;
  mini?: boolean;
};

const formatPct = (v: number) => `${Math.round(v * 100)}%`;

const ParamControlNetBeginEnd = (props: Props) => {
  const { controlNetId, beginStepPct, endStepPct, mini = false } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleStepPctChanged = useCallback(
    (v: number[]) => {
      dispatch(
        controlNetBeginStepPctChanged({ controlNetId, beginStepPct: v[0] })
      );
      dispatch(controlNetEndStepPctChanged({ controlNetId, endStepPct: v[1] }));
    },
    [controlNetId, dispatch]
  );

  const handleStepPctReset = useCallback(() => {
    dispatch(controlNetBeginStepPctChanged({ controlNetId, beginStepPct: 0 }));
    dispatch(controlNetEndStepPctChanged({ controlNetId, endStepPct: 1 }));
  }, [controlNetId, dispatch]);

  return (
    <FormControl>
      <FormLabel>Begin / End Step Percentage</FormLabel>
      <HStack w="100%" gap={2} alignItems="center">
        <RangeSlider
          aria-label={['Begin Step %', 'End Step %']}
          value={[beginStepPct, endStepPct]}
          onChange={handleStepPctChanged}
          min={0}
          max={1}
          step={0.01}
          minStepsBetweenThumbs={5}
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
          {!mini && (
            <>
              <RangeSliderMark
                value={0}
                sx={{
                  fontSize: 'xs',
                  fontWeight: '500',
                  color: 'base.200',
                  insetInlineStart: '0 !important',
                  insetInlineEnd: 'unset !important',
                  mt: 1.5,
                }}
              >
                0%
              </RangeSliderMark>
              <RangeSliderMark
                value={0.5}
                sx={{
                  fontSize: 'xs',
                  fontWeight: '500',
                  color: 'base.200',
                  mt: 1.5,
                }}
              >
                50%
              </RangeSliderMark>
              <RangeSliderMark
                value={1}
                sx={{
                  fontSize: 'xs',
                  fontWeight: '500',
                  color: 'base.200',
                  insetInlineStart: 'unset !important',
                  insetInlineEnd: '0 !important',
                  mt: 1.5,
                }}
              >
                100%
              </RangeSliderMark>
            </>
          )}
        </RangeSlider>

        {!mini && (
          <IAIIconButton
            size="sm"
            aria-label={t('accessibility.reset')}
            tooltip={t('accessibility.reset')}
            icon={<BiReset />}
            onClick={handleStepPctReset}
          />
        )}
      </HStack>
    </FormControl>
  );
};

export default memo(ParamControlNetBeginEnd);
