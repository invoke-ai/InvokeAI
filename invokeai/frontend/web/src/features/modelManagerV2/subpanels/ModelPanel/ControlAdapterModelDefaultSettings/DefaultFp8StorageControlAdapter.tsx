import { Flex, FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { ControlAdapterModelDefaultSettingsFormData } from './ControlAdapterModelDefaultSettings';

type DefaultFp8StorageType = ControlAdapterModelDefaultSettingsFormData['fp8Storage'];

export const DefaultFp8StorageControlAdapter = memo(
  (props: UseControllerProps<ControlAdapterModelDefaultSettingsFormData>) => {
    const { t } = useTranslation();
    const { field } = useController(props);

    const onChange = useCallback(
      (e: ChangeEvent<HTMLInputElement>) => {
        const updatedValue = {
          ...(field.value as DefaultFp8StorageType),
          value: e.target.checked,
        };
        field.onChange(updatedValue);
      },
      [field]
    );

    const value = useMemo(() => {
      return (field.value as DefaultFp8StorageType).value;
    }, [field.value]);

    const isDisabled = useMemo(() => {
      return !(field.value as DefaultFp8StorageType).isEnabled;
    }, [field.value]);

    return (
      <FormControl flexDir="column" gap={1} alignItems="flex-start">
        <Flex justifyContent="space-between" w="full">
          <InformationalPopover feature="fp8Storage">
            <FormLabel>{t('modelManager.fp8Storage')}</FormLabel>
          </InformationalPopover>
          <SettingToggle control={props.control} name="fp8Storage" />
        </Flex>
        <Switch isChecked={value} onChange={onChange} isDisabled={isDisabled} />
      </FormControl>
    );
  }
);

DefaultFp8StorageControlAdapter.displayName = 'DefaultFp8StorageControlAdapter';
