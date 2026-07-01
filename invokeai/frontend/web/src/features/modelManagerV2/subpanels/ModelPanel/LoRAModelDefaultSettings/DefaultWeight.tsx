import { CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { DEFAULT_LORA_WEIGHT_CONFIG } from 'features/controlLayers/store/lorasSlice';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController, useWatch } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { LoRAModelDefaultSettingsFormData } from './LoRAModelDefaultSettings';

export const DefaultWeight = memo((props: UseControllerProps<LoRAModelDefaultSettingsFormData, 'weight'>) => {
  const { field } = useController(props);
  const { t } = useTranslation();

  const weightMin = useWatch({ control: props.control, name: 'weightMin' });
  const weightMax = useWatch({ control: props.control, name: 'weightMax' });

  const sliderMin = weightMin?.isEnabled ? weightMin.value : DEFAULT_LORA_WEIGHT_CONFIG.sliderMin;
  const sliderMax = weightMax?.isEnabled ? weightMax.value : DEFAULT_LORA_WEIGHT_CONFIG.sliderMax;
  const numberInputMin = Math.min(sliderMin, DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin);
  const numberInputMax = Math.max(sliderMax, DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax);

  const onChange = useCallback(
    (v: number) => {
      const updatedValue = {
        ...field.value,
        value: v,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return field.value.value;
  }, [field.value]);

  const isDisabled = useMemo(() => {
    return !field.value.isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={2} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <InformationalPopover feature="loraWeight">
          <FormLabel>{t('lora.startingWeight')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={props.control} name="weight" />
      </Flex>

      <Flex w="full" gap={4}>
        <CompositeNumberInput
          value={value}
          min={numberInputMin}
          max={numberInputMax}
          step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
          fineStep={DEFAULT_LORA_WEIGHT_CONFIG.fineStep}
          onChange={onChange}
          isDisabled={isDisabled}
        />
      </Flex>
    </FormControl>
  );
});

DefaultWeight.displayName = 'DefaultWeight';
