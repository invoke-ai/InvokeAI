import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { MainModelDefaultSettingsFormData } from './MainModelDefaultSettings';

type DefaultFp8StorageType = MainModelDefaultSettingsFormData['fp8Storage'];

export const DefaultFp8Storage = memo((props: UseControllerProps<MainModelDefaultSettingsFormData>) => {
  const { t } = useTranslation();
  const { field } = useController(props);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const updatedValue = {
        ...(field.value as DefaultFp8StorageType),
        value: e.target.checked,
        isEnabled: e.target.checked,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return (field.value as DefaultFp8StorageType).value;
  }, [field.value]);

  return (
    <FormControl>
      <InformationalPopover feature="fp8Storage">
        <FormLabel>{t('modelManager.fp8Storage')}</FormLabel>
      </InformationalPopover>
      <Switch isChecked={value} onChange={onChange} />
    </FormControl>
  );
});

DefaultFp8Storage.displayName = 'DefaultFp8Storage';
