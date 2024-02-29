import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { DefaultSettingsFormData } from '../DefaultSettings';
import { UseControllerProps, useController } from 'react-hook-form';
import { useGetModelConfigQuery, useGetVaeModelsQuery } from '../../../../../services/api/endpoints/models';
import { map } from 'lodash-es';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from '../../../../../app/store/storeHooks';

type DefaultVaeType = DefaultSettingsFormData['vae'];

export function DefaultVae(props: UseControllerProps<DefaultSettingsFormData>) {
  const { t } = useTranslation();
  const { field } = useController(props);
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data: modelData } = useGetModelConfigQuery(selectedModelKey ?? skipToken);

  const { compatibleOptions } = useGetVaeModelsQuery(undefined, {
    selectFromResult: ({ data }) => {
      const modelArray = map(data?.entities);
      const compatibleOptions = modelArray
        .filter((vae) => vae.base === modelData?.base)
        .map((vae) => ({ label: vae.name, value: vae.key }));

      return { compatibleOptions };
    },
  });

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      const newValue = !v?.value ? null : v.value;

      const updatedValue = {
        ...(field.value as DefaultVaeType),
        value: newValue,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return compatibleOptions.find((vae) => vae.value === (field.value as DefaultVaeType).value);
  }, [compatibleOptions, field.value]);

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultVaeType).isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={1} alignItems="flex-start">
      <InformationalPopover feature="paramVAE">
        <FormLabel>{t('modelManager.vae')}</FormLabel>
      </InformationalPopover>
      <Combobox
        isDisabled={isDisabled}
        isClearable
        value={value}
        placeholder={value ? value.value : t('models.defaultVAE')}
        options={compatibleOptions}
        onChange={onChange}
      />
    </FormControl>
  );
}
