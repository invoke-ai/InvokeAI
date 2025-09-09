import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { DEFAULT_LORA_WEIGHT_CONFIG } from 'features/controlLayers/store/lorasSlice';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { LoRAModelDefaultSettingsFormData } from './LoRAModelDefaultSettings';

const MARKS = [-1, 0, 1, 2];

type DefaultWeight = LoRAModelDefaultSettingsFormData['weight'];

export const DefaultWeight = memo((props: UseControllerProps<LoRAModelDefaultSettingsFormData, 'weight'>) => {
  const { field } = useController(props);
  const { t } = useTranslation();

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
          <FormLabel>{t('lora.weight')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={props.control} name="weight" />
      </Flex>

      <Flex w="full" gap={4}>
        <CompositeSlider
          value={value}
          min={DEFAULT_LORA_WEIGHT_CONFIG.sliderMin}
          max={DEFAULT_LORA_WEIGHT_CONFIG.sliderMax}
          step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
          fineStep={DEFAULT_LORA_WEIGHT_CONFIG.fineStep}
          onChange={onChange}
          marks={MARKS}
          isDisabled={isDisabled}
        />
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

DefaultWeight.displayName = 'DefaultWeight';
