import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { selectLoRAWeightConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { LoRAModelDefaultSettingsFormData } from './LoRAModelDefaultSettings';

type DefaultWeight = LoRAModelDefaultSettingsFormData['weight'];

export const DefaultWeight = memo((props: UseControllerProps<LoRAModelDefaultSettingsFormData>) => {
  const { field } = useController(props);

  const config = useAppSelector(selectLoRAWeightConfig);
  const { t } = useTranslation();
  const marks = useMemo(
    () => [config.sliderMin, Math.floor(config.sliderMax / 2), config.sliderMax],
    [config.sliderMax, config.sliderMin]
  );

  const onChange = useCallback(
    (v: number) => {
      const updatedValue = {
        ...(field.value as DefaultWeight),
        value: v,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return (field.value as DefaultWeight).value;
  }, [field.value]);

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultWeight).isEnabled;
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
          min={config.sliderMin}
          max={config.sliderMax}
          step={config.coarseStep}
          fineStep={config.fineStep}
          onChange={onChange}
          marks={marks}
          isDisabled={isDisabled}
        />
        <CompositeNumberInput
          value={value}
          min={config.numberInputMin}
          max={config.numberInputMax}
          step={config.coarseStep}
          fineStep={config.fineStep}
          onChange={onChange}
          isDisabled={isDisabled}
        />
      </Flex>
    </FormControl>
  );
});

DefaultWeight.displayName = 'DefaultWeight';
