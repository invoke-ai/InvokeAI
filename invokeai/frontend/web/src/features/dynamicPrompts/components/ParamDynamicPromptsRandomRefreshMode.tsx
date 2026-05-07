import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isDynamicPromptRandomRefreshMode,
  randomRefreshModeChanged,
  selectDynamicPromptsMode,
  selectDynamicPromptsRandomRefreshMode,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback, useMemo } from 'react';

const ParamDynamicPromptsRandomRefreshMode = () => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectDynamicPromptsMode);
  const randomRefreshMode = useAppSelector(selectDynamicPromptsRandomRefreshMode);

  const options = useMemo<ComboboxOption[]>(
    () => [
      {
        value: 'per_image',
        label: 'Per Image',
        description: 'Roll a new random sample for each generated image.',
      },
      {
        value: 'per_enqueue',
        label: 'Per Invoke',
        description: 'Random wildcards roll once per Invoke; cyclic wildcards still advance per queued output.',
      },
      {
        value: 'manual',
        label: 'Locked Preview',
        description: 'Keep the preview fixed until Reshuffle is used.',
      },
    ],
    []
  );

  const handleChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isDynamicPromptRandomRefreshMode(v?.value)) {
        return;
      }
      dispatch(randomRefreshModeChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === randomRefreshMode), [options, randomRefreshMode]);

  if (mode !== 'random') {
    return null;
  }

  return (
    <FormControl>
      <FormLabel>Randomness</FormLabel>
      <Combobox value={value} options={options} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsRandomRefreshMode);
