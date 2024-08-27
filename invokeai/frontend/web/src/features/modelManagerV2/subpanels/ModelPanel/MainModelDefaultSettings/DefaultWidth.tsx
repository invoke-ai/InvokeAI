import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { selectWidthConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { MainModelDefaultSettingsFormData } from './MainModelDefaultSettings';

type DefaultWidthType = MainModelDefaultSettingsFormData['width'];

type Props = {
  control: UseControllerProps<MainModelDefaultSettingsFormData>['control'];
  optimalDimension: number;
};

export const DefaultWidth = memo(({ control, optimalDimension }: Props) => {
  const { field } = useController({ control, name: 'width' });
  const config = useAppSelector(selectWidthConfig);
  const { t } = useTranslation();
  const marks = useMemo(
    () => [config.sliderMin, optimalDimension, config.sliderMax],
    [config.sliderMin, optimalDimension, config.sliderMax]
  );

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

DefaultWidth.displayName = 'DefaultWidth';
