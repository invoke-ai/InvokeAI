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
        value: 'per_enqueue',
        label: 'Every Invoke',
        description: 'Roll a new random sample when generation is queued.',
      },
      {
        value: 'manual',
        label: 'Manual',
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
      <FormLabel>Refresh</FormLabel>
      <Combobox value={value} options={options} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsRandomRefreshMode);
