import {
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormLabel,
  Radio,
  RadioGroup,
  Text,
} from '@invoke-ai/ui-library';
import type { AlphaToOutlineFilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<AlphaToOutlineFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.alpha_to_outline.buildDefaults();

export const FilterAlphaToOutline = ({ onChange, config }: Props) => {
  const { t } = useTranslation();
  const handleLineWidthPercentChange = useCallback(
    (v: number) => {
      onChange({ ...config, line_width_percent: v });
    },
    [onChange, config]
  );

  const handleLineModeChange = useCallback(
    (v: string) => {
      onChange({ ...config, line_mode: v });
    },
    [onChange, config]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.alpha_to_outline.line_width_percent')}</FormLabel>
        <CompositeSlider
          value={config.line_width_percent}
          onChange={handleLineWidthPercentChange}
          defaultValue={DEFAULTS.line_width_percent}
          min={1}
          max={100}
        />
        <CompositeNumberInput
          value={config.line_width_percent}
          onChange={handleLineWidthPercentChange}
          defaultValue={DEFAULTS.line_width_percent}
          min={1}
          max={100}
        />
      </FormControl>
      <FormControl w="min-content">
        <FormLabel m={0}>{t('controlLayers.filter.alpha_to_outline.line_mode')}</FormLabel>
        <RadioGroup value={config.line_mode} onChange={handleLineModeChange} w="full" size="md">
          <Flex alignItems="center" w="full" gap={4} fontWeight="semibold" color="base.300">
            <Radio value="both">
              <Text>{t('controlLayers.filter.alpha_to_outline.both')}</Text>
            </Radio>
            <Radio value="inner">
              <Text>{t('controlLayers.filter.alpha_to_outline.inner')}</Text>
            </Radio>
            <Radio value="outer">
              <Text>{t('controlLayers.filter.alpha_to_outline.outer')}</Text>
            </Radio>
          </Flex>
        </RadioGroup>
      </FormControl>
    </>
  );
};

FilterAlphaToOutline.displayName = 'FilterAlphaToOutline';
