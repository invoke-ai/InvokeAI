import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { CONSTRAINTS } from 'features/parameters/components/Advanced/ParamCFGRescaleMultiplier';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { MainModelDefaultSettingsFormData } from './MainModelDefaultSettings';

type DefaultCfgRescaleMultiplierType = MainModelDefaultSettingsFormData['cfgRescaleMultiplier'];

export const DefaultCfgRescaleMultiplier = memo((props: UseControllerProps<MainModelDefaultSettingsFormData>) => {
  const { field } = useController(props);

  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      const updatedValue = {
        ...(field.value as DefaultCfgRescaleMultiplierType),
        value: v,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return (field.value as DefaultCfgRescaleMultiplierType).value;
  }, [field.value]);

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultCfgRescaleMultiplierType).isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={2} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <InformationalPopover feature="paramCFGRescaleMultiplier">
          <FormLabel>{t('parameters.cfgRescaleMultiplier')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={props.control} name="cfgRescaleMultiplier" />
      </Flex>

      <Flex w="full" gap={4}>
        <CompositeSlider
          value={value}
          min={CONSTRAINTS.sliderMin}
          max={CONSTRAINTS.sliderMax}
          step={CONSTRAINTS.coarseStep}
          fineStep={CONSTRAINTS.fineStep}
          onChange={onChange}
          isDisabled={isDisabled}
          marks
        />
        <CompositeNumberInput
          value={value}
          min={CONSTRAINTS.numberInputMin}
          max={CONSTRAINTS.numberInputMax}
          step={CONSTRAINTS.coarseStep}
          fineStep={CONSTRAINTS.fineStep}
          onChange={onChange}
          isDisabled={isDisabled}
        />
      </Flex>
    </FormControl>
  );
});

DefaultCfgRescaleMultiplier.displayName = 'DefaultCfgRescaleMultiplier';
