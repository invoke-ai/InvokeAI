import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isDynamicPromptMode,
  modeChanged,
  selectDynamicPromptsMode,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback, useMemo } from 'react';

const ParamDynamicPromptsMode = () => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectDynamicPromptsMode);

  const options = useMemo<ComboboxOption[]>(
    () => [
      {
        value: 'random',
        label: 'Random Sample',
        description: 'Sample prompts. Randomness applies to random wildcards.',
      },
      {
        value: 'combinatorial',
        label: 'All Combinations',
        description: 'Preview and queue every combination up to the limit.',
      },
    ],
    []
  );

  const handleChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isDynamicPromptMode(v?.value)) {
        return;
      }
      dispatch(modeChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === mode), [mode, options]);

  return (
    <FormControl>
      <FormLabel>Mode</FormLabel>
      <Combobox value={value} options={options} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsMode);
