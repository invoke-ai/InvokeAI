import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { selectGuidanceConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { MainModelDefaultSettingsFormData } from './MainModelDefaultSettings';

type DefaultGuidanceType = MainModelDefaultSettingsFormData['guidance'];

export const DefaultGuidance = memo((props: UseControllerProps<MainModelDefaultSettingsFormData>) => {
  const { field } = useController(props);

  const config = useAppSelector(selectGuidanceConfig);
  const { t } = useTranslation();
  const marks = useMemo(
    () => [
      config.sliderMin,
      Math.floor(config.sliderMax - (config.sliderMax - config.sliderMin) / 2),
      config.sliderMax,
    ],
    [config.sliderMax, config.sliderMin]
  );

  const onChange = useCallback(
    (v: number) => {
      const updatedValue = {
        ...(field.value as DefaultGuidanceType),
        value: v,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return (field.value as DefaultGuidanceType).value;
  }, [field.value]);

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultGuidanceType).isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={2} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <InformationalPopover feature="paramGuidance">
          <FormLabel>{t('parameters.guidance')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={props.control} name="guidance" />
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

DefaultGuidance.displayName = 'DefaultGuidance';
