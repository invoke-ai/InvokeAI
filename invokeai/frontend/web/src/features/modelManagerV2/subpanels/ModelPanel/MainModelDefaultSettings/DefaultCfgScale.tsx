import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { MainModelDefaultSettingsFormData } from './MainModelDefaultSettings';

type DefaultCfgType = MainModelDefaultSettingsFormData['cfgScale'];

export function DefaultCfgScale(props: UseControllerProps<MainModelDefaultSettingsFormData>) {
  const { field } = useController(props);

  const sliderMin = useAppSelector((s) => s.config.sd.guidance.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.guidance.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.guidance.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.guidance.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.guidance.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.guidance.fineStep);
  const { t } = useTranslation();
  const marks = useMemo(() => [sliderMin, Math.floor(sliderMax / 2), sliderMax], [sliderMax, sliderMin]);

  const onChange = useCallback(
    (v: number) => {
      const updatedValue = {
        ...(field.value as DefaultCfgType),
        value: v,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return (field.value as DefaultCfgType).value;
  }, [field.value]);

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultCfgType).isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={2} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <InformationalPopover feature="paramCFGScale">
          <FormLabel>{t('parameters.cfgScale')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={props.control} name="cfgScale" />
      </Flex>

      <Flex w="full" gap={4}>
        <CompositeSlider
          value={value}
          min={sliderMin}
          max={sliderMax}
          step={coarseStep}
          fineStep={fineStep}
          onChange={onChange}
          marks={marks}
          isDisabled={isDisabled}
        />
        <CompositeNumberInput
          value={value}
          min={numberInputMin}
          max={numberInputMax}
          step={coarseStep}
          fineStep={fineStep}
          onChange={onChange}
          isDisabled={isDisabled}
        />
      </Flex>
    </FormControl>
  );
}
