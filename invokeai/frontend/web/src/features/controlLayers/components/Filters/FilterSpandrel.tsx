import {
  Box,
  Combobox,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormHelperText,
  FormLabel,
  Switch,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import type { SpandrelFilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import type { ChangeEvent } from 'react';
import { useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useSpandrelImageToImageModels } from 'services/api/hooks/modelsByType';
import type { SpandrelImageToImageModelConfig } from 'services/api/types';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<SpandrelFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.spandrel_filter.buildDefaults();

export const FilterSpandrel = ({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const [modelConfigs, { isLoading }] = useSpandrelImageToImageModels();

  const tooltipLabel = useMemo(() => {
    if (!modelConfigs.length || !config.model) {
      return;
    }
    return modelConfigs.find((m) => m.key === config.model?.key)?.description;
  }, [modelConfigs, config.model]);

  const _onChange = useCallback(
    (v: SpandrelImageToImageModelConfig | null) => {
      onChange({ ...config, model: v });
    },
    [config, onChange]
  );

  const {
    options,
    value,
    onChange: onChangeModel,
    placeholder,
    noOptionsMessage,
  } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: config.model,
    isLoading,
  });

  const onScaleChanged = useCallback(
    (v: number) => {
      onChange({ ...config, scale: v });
    },
    [onChange, config]
  );
  const onAutoscaleChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, autoScale: e.target.checked });
    },
    [onChange, config]
  );

  useEffect(() => {
    const firstModel = options[0];
    if (!config.model && firstModel) {
      onChangeModel(firstModel);
    }
  }, [config.model, onChangeModel, options]);

  return (
    <>
      <FormControl w="full" orientation="vertical">
        <Flex w="full" alignItems="center">
          <FormLabel m={0} flexGrow={1}>
            {t('controlLayers.filter.spandrel_filter.autoScale')}
          </FormLabel>
          <Switch size="sm" isChecked={config.autoScale} onChange={onAutoscaleChanged} />
        </Flex>
        <FormHelperText>{t('controlLayers.filter.spandrel_filter.autoScaleDesc')}</FormHelperText>
      </FormControl>
      <FormControl isDisabled={!config.autoScale}>
        <FormLabel m={0}>{t('controlLayers.filter.spandrel_filter.scale')}</FormLabel>
        <CompositeSlider
          value={config.scale}
          onChange={onScaleChanged}
          defaultValue={DEFAULTS.scale}
          min={1}
          max={16}
        />
        <CompositeNumberInput
          value={config.scale}
          onChange={onScaleChanged}
          defaultValue={DEFAULTS.scale}
          min={1}
          max={16}
        />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.spandrel_filter.model')}</FormLabel>
        <Tooltip label={tooltipLabel}>
          <Box w="full">
            <Combobox
              value={value}
              placeholder={placeholder}
              options={options}
              onChange={onChangeModel}
              noOptionsMessage={noOptionsMessage}
              isDisabled={options.length === 0}
            />
          </Box>
        </Tooltip>
      </FormControl>
    </>
  );
};

FilterSpandrel.displayName = 'FilterSpandrel';
