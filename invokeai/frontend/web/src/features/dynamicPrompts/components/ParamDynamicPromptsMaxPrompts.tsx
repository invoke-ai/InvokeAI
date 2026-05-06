import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  maxCombinationsChanged,
  randomSamplesChanged,
  selectDynamicPromptsMaxCombinations,
  selectDynamicPromptsMode,
  selectDynamicPromptsRandomSamples,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback, useMemo } from 'react';

const CONSTRAINTS = {
  initial: 100,
  sliderMin: 1,
  sliderMax: 1000,
  numberInputMin: 1,
  numberInputMax: 10000,
  fineStep: 1,
  coarseStep: 10,
};

const ParamDynamicPromptsMaxPrompts = () => {
  const mode = useAppSelector(selectDynamicPromptsMode);
  const randomSamples = useAppSelector(selectDynamicPromptsRandomSamples);
  const maxCombinations = useAppSelector(selectDynamicPromptsMaxCombinations);
  const dispatch = useAppDispatch();
  const value = mode === 'combinatorial' ? maxCombinations : randomSamples;
  const label = useMemo(() => (mode === 'combinatorial' ? 'Max Combinations' : 'Random Samples'), [mode]);

  const handleChange = useCallback(
    (v: number) => {
      if (mode === 'combinatorial') {
        dispatch(maxCombinationsChanged(v));
      } else {
        dispatch(randomSamplesChanged(v));
      }
    },
    [dispatch, mode]
  );

  return (
    <FormControl>
      <InformationalPopover feature="dynamicPromptsMaxPrompts" inPortal={false}>
        <FormLabel>{label}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        value={value}
        defaultValue={mode === 'combinatorial' ? CONSTRAINTS.initial : 1}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        value={value}
        defaultValue={mode === 'combinatorial' ? CONSTRAINTS.initial : 1}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsMaxPrompts);
