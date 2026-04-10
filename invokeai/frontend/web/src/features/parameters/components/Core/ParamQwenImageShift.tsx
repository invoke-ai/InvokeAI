import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { qwenImageShiftChanged, selectQwenImageShift } from 'features/controlLayers/store/paramsSlice';
import type React from 'react';
import { memo, useCallback } from 'react';
import { PiXBold } from 'react-icons/pi';

const CONSTRAINTS = {
  initial: 3,
  sliderMin: 1,
  sliderMax: 7,
  numberInputMin: 0,
  numberInputMax: 10,
  fineStep: 0.1,
  coarseStep: 0.5,
};

const MARKS = [1, 2, 3, 4, 5, 6, 7];

const ParamQwenImageShift = () => {
  const shift = useAppSelector(selectQwenImageShift);
  const dispatch = useAppDispatch();

  const onChange = useCallback((v: number) => dispatch(qwenImageShiftChanged(v)), [dispatch]);
  const onReset = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dispatch(qwenImageShiftChanged(null));
    },
    [dispatch]
  );

  const displayValue = shift ?? CONSTRAINTS.initial;

  return (
    <FormControl>
      <FormLabel>
        Shift{' '}
        {shift !== null ? (
          <Text as="span" cursor="pointer" onClick={onReset} display="inline-flex" verticalAlign="middle">
            <PiXBold />
          </Text>
        ) : (
          <Text as="span" opacity={0.5} fontWeight="normal" fontSize="xs">
            (auto)
          </Text>
        )}
      </FormLabel>
      <CompositeSlider
        value={displayValue}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={displayValue}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamQwenImageShift);
