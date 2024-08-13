import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { StylePresetFormData } from './StylePresetForm';

const OPTIONS = [
  { label: t('stylePresets.private'), value: 'user' },
  { label: t('stylePresets.shared'), value: 'project' },
];

export const StylePresetTypeField = (props: UseControllerProps<StylePresetFormData, 'type'>) => {
  const { field } = useController(props);
  const stylePresetModalState = useStore($stylePresetModalState);
  const { t } = useTranslation();

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (v) {
        field.onChange(v.value);
      }
    },
    [field]
  );

  const value = useMemo(() => {
    return OPTIONS.find((opt) => opt.value === field.value);
  }, [field.value]);

  return (
    <FormControl
      orientation="vertical"
      maxW={48}
      isDisabled={stylePresetModalState.prefilledFormData?.type === 'project'}
    >
      <FormLabel>{t('stylePresets.type')}</FormLabel>
      <Combobox value={value} options={OPTIONS} onChange={onChange} />
    </FormControl>
  );
};
