import {
  forwardRef,
  RangeSlider as ChakraRangeSlider,
  RangeSliderFilledTrack as ChakraRangeSliderFilledTrack,
  RangeSliderThumb as ChakraRangeSliderThumb,
  RangeSliderTrack as ChakraRangeSliderTrack,
  useFormControl,
} from '@chakra-ui/react';
import { useStore } from '@nanostores/react';
import type { InvFormattedMark } from 'common/components/InvSlider/types';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { $shift } from 'common/hooks/useGlobalModifiers';
import { AnimatePresence } from 'framer-motion';
import { memo, useCallback, useMemo, useState } from 'react';

import { InvRangeSliderMark } from './InvRangeSliderMark';
import type { InvRangeSliderProps } from './types';

export const InvRangeSlider = memo(
  forwardRef<InvRangeSliderProps, typeof ChakraRangeSlider>(
    (props: InvRangeSliderProps, ref) => {
      const {
        value,
        min,
        max,
        step: _step = 1,
        fineStep: _fineStep,
        onChange,
        onReset,
        formatValue = (v: number) => v.toString(),
        marks: _marks,
        withThumbTooltip: withTooltip = false,
        ...sliderProps
      } = props;
      const [isMouseOverSlider, setIsMouseOverSlider] = useState(false);
      const [isChanging, setIsChanging] = useState(false);

      const shift = useStore($shift);
      const step = useMemo(
        () => (shift ? _fineStep ?? _step : _step),
        [shift, _fineStep, _step]
      );
      const controlProps = useFormControl({});

      const labels = useMemo<string[]>(
        () => value.map(formatValue),
        [formatValue, value]
      );

      const onMouseEnter = useCallback(() => setIsMouseOverSlider(true), []);
      const onMouseLeave = useCallback(() => setIsMouseOverSlider(false), []);
      const onChangeStart = useCallback(() => setIsChanging(true), []);
      const onChangeEnd = useCallback(() => setIsChanging(false), []);

      const marks = useMemo<InvFormattedMark[]>(() => {
        if (_marks === true) {
          return [min, max].map((m) => ({ value: m, label: formatValue(m) }));
        }
        if (_marks) {
          return _marks?.map((m) => ({ value: m, label: formatValue(m) }));
        }
        return [];
      }, [_marks, formatValue, max, min]);

      return (
        <ChakraRangeSlider
          ref={ref}
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
          onMouseEnter={onMouseEnter}
          onMouseLeave={onMouseLeave}
          focusThumbOnChange={false}
          onChangeStart={onChangeStart}
          onChangeEnd={onChangeEnd}
          {...sliderProps}
          {...controlProps}
        >
          <AnimatePresence>
            {marks?.length &&
              (isMouseOverSlider || isChanging) &&
              marks.map((m, i) => (
                <InvRangeSliderMark
                  key={m.value}
                  value={m.value}
                  label={m.label}
                  index={i}
                  total={marks.length}
                />
              ))}
          </AnimatePresence>

          <ChakraRangeSliderTrack>
            <ChakraRangeSliderFilledTrack />
          </ChakraRangeSliderTrack>

          <InvTooltip
            isOpen={withTooltip && (isMouseOverSlider || isChanging)}
            label={labels[0]}
          >
            <ChakraRangeSliderThumb
              index={0}
              onDoubleClick={onReset}
              zIndex={0}
            />
          </InvTooltip>
          <InvTooltip
            isOpen={withTooltip && (isMouseOverSlider || isChanging)}
            label={labels[1]}
          >
            <ChakraRangeSliderThumb
              index={1}
              onDoubleClick={onReset}
              zIndex={0}
            />
          </InvTooltip>
        </ChakraRangeSlider>
      );
    }
  )
);
