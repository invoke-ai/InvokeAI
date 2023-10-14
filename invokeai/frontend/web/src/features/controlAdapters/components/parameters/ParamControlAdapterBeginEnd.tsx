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
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import { useControlAdapterBeginEndStepPct } from 'features/controlAdapters/hooks/useControlAdapterBeginEndStepPct';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import {
  controlAdapterBeginStepPctChanged,
  controlAdapterEndStepPctChanged,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

const formatPct = (v: number) => `${Math.round(v * 100)}%`;

const ParamControlAdapterBeginEnd = ({ id }: Props) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const stepPcts = useControlAdapterBeginEndStepPct(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleStepPctChanged = useCallback(
    (v: number[]) => {
      dispatch(
        controlAdapterBeginStepPctChanged({
          id,
          beginStepPct: v[0] as number,
        })
      );
      dispatch(
        controlAdapterEndStepPctChanged({
          id,
          endStepPct: v[1] as number,
        })
      );
    },
    [dispatch, id]
  );

  if (!stepPcts) {
    return null;
  }

  return (
    <IAIInformationalPopover feature="controlNetBeginEnd">
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.beginEndStepPercent')}</FormLabel>
        <HStack w="100%" gap={2} alignItems="center">
          <RangeSlider
            aria-label={['Begin Step %', 'End Step %!']}
            value={[stepPcts.beginStepPct, stepPcts.endStepPct]}
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
            <Tooltip
              label={formatPct(stepPcts.beginStepPct)}
              placement="top"
              hasArrow
            >
              <RangeSliderThumb index={0} />
            </Tooltip>
            <Tooltip
              label={formatPct(stepPcts.endStepPct)}
              placement="top"
              hasArrow
            >
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
    </IAIInformationalPopover>
  );
};

export default memo(ParamControlAdapterBeginEnd);
