import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { MainModelDefaultSettingsFormData } from './MainModelDefaultSettings';

type DefaultHeightType = MainModelDefaultSettingsFormData['height'];

type Props = {
  control: UseControllerProps<MainModelDefaultSettingsFormData>['control'];
  optimalDimension: number;
};

export function DefaultHeight({ control, optimalDimension }: Props) {
  const { field } = useController({ control, name: 'height' });
  const sliderMin = useAppSelector((s) => s.config.sd.height.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.height.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.height.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.height.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.height.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.height.fineStep);
  const { t } = useTranslation();
  const marks = useMemo(() => [sliderMin, optimalDimension, sliderMax], [sliderMin, optimalDimension, sliderMax]);

  const onChange = useCallback(
    (v: number) => {
      const updatedValue = {
        ...(field.value as DefaultHeightType),
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
        <InformationalPopover feature="paramHeight">
          <FormLabel>{t('parameters.height')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={control} name="height" />
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
