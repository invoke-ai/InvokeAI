import { CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { DEFAULT_LORA_WEIGHT_CONFIG } from 'features/controlLayers/store/lorasSlice';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { LoRAModelDefaultSettingsFormData } from './LoRAModelDefaultSettings';

export const DefaultWeightMax = memo((props: UseControllerProps<LoRAModelDefaultSettingsFormData, 'weightMax'>) => {
  const { field } = useController(props);
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      field.onChange({ ...field.value, value: v });
    },
    [field]
  );

  const value = useMemo(() => field.value.value, [field.value]);
  const isDisabled = useMemo(() => !field.value.isEnabled, [field.value]);

  return (
    <FormControl flexDir="column" gap={2} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <FormLabel>{t('lora.weightMax')}</FormLabel>
        <SettingToggle control={props.control} name="weightMax" />
      </Flex>
      <Flex w="full" gap={4}>
        <CompositeNumberInput
          value={value}
          min={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
          max={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
          step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
          fineStep={DEFAULT_LORA_WEIGHT_CONFIG.fineStep}
          onChange={onChange}
          isDisabled={isDisabled}
        />
      </Flex>
    </FormControl>
  );
});

DefaultWeightMax.displayName = 'DefaultWeightMax';
