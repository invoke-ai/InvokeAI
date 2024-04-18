import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isParameterAutoNegative } from 'features/parameters/types/parameterSchemas';
import { autoNegativeChanged } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const options: ComboboxOption[] = [
  { label: 'Off', value: 'off' },
  { label: 'Invert', value: 'invert' },
];

const AutoNegativeCombobox = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const autoNegative = useAppSelector((s) => s.regionalPrompts.present.autoNegative);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterAutoNegative(v?.value)) {
        return;
      }
      dispatch(autoNegativeChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === autoNegative), [autoNegative]);

  return (
    <FormControl>
      <FormLabel>Negative Mode</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} isSearchable={false} />
    </FormControl>
  );
};

export default memo(AutoNegativeCombobox);
