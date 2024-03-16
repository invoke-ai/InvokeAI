import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { MainModelDefaultSettingsFormData } from './MainModelDefaultSettings';

type DefaultWidthType = MainModelDefaultSettingsFormData['width'];

type Props = {
  control: UseControllerProps<MainModelDefaultSettingsFormData>['control'];
  optimalDimension: number;
};

export function DefaultWidth({ control, optimalDimension }: Props) {
  const { field } = useController({ control, name: 'width' });
  const sliderMin = useAppSelector((s) => s.config.sd.width.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.width.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.width.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.width.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.width.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.width.fineStep);
  const { t } = useTranslation();
  const marks = useMemo(() => [sliderMin, optimalDimension, sliderMax], [sliderMin, optimalDimension, sliderMax]);

  const onChange = useCallback(
    (v: number) => {
      const updatedValue = {
        ...(field.value as DefaultWidthType),
        value: v,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return (field.value as DefaultWidthType).value;
  }, [field.value]);

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultWidthType).isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={2} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <InformationalPopover feature="paramWidth">
          <FormLabel>{t('parameters.width')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={control} name="width" />
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
