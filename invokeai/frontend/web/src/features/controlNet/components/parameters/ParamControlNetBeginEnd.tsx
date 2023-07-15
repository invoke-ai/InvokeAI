import {
  ChakraProps,
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
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import {
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
} from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback, useMemo } from 'react';

const SLIDER_MARK_STYLES: ChakraProps['sx'] = {
  mt: 1.5,
  fontSize: '2xs',
  fontWeight: '500',
  color: 'base.400',
};

type Props = {
  controlNetId: string;
  mini?: boolean;
};

const formatPct = (v: number) => `${Math.round(v * 100)}%`;

const ParamControlNetBeginEnd = (props: Props) => {
  const { controlNetId, mini = false } = props;
  const dispatch = useAppDispatch();

  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlNet }) => {
          const { beginStepPct, endStepPct } =
            controlNet.controlNets[controlNetId];
          return { beginStepPct, endStepPct };
        },
        defaultSelectorOptions
      ),
    [controlNetId]
  );

  const { beginStepPct, endStepPct } = useAppSelector(selector);

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
                  insetInlineStart: '0 !important',
                  insetInlineEnd: 'unset !important',
                  ...SLIDER_MARK_STYLES,
                }}
              >
                0%
              </RangeSliderMark>
              <RangeSliderMark
                value={0.5}
                sx={{
                  ...SLIDER_MARK_STYLES,
                }}
              >
                50%
              </RangeSliderMark>
              <RangeSliderMark
                value={1}
                sx={{
                  insetInlineStart: 'unset !important',
                  insetInlineEnd: '0 !important',
                  ...SLIDER_MARK_STYLES,
                }}
              >
                100%
              </RangeSliderMark>
            </>
          )}
        </RangeSlider>
      </HStack>
    </FormControl>
  );
};

export default memo(ParamControlNetBeginEnd);
