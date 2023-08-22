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
import {
  ControlNetConfig,
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
} from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';

type Props = {
  controlNet: ControlNetConfig;
};

const formatPct = (v: number) => `${Math.round(v * 100)}%`;

const ParamControlNetBeginEnd = (props: Props) => {
  const { beginStepPct, endStepPct, isEnabled, controlNetId } =
    props.controlNet;
  const dispatch = useAppDispatch();

  const handleStepPctChanged = useCallback(
    (v: number[]) => {
      dispatch(
        controlNetBeginStepPctChanged({
          controlNetId,
          beginStepPct: v[0] as number,
        })
      );
      dispatch(
        controlNetEndStepPctChanged({
          controlNetId,
          endStepPct: v[1] as number,
        })
      );
    },
    [controlNetId, dispatch]
  );

  return (
    <FormControl isDisabled={!isEnabled}>
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

export default memo(ParamControlNetBeginEnd);
