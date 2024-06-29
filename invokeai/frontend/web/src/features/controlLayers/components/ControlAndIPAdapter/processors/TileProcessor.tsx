import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { isTileProcessorMode, type TileProcessorMode } from 'features/controlAdapters/store/types';
import type { ProcessorComponentProps } from 'features/controlLayers/components/ControlAndIPAdapter/processors/types';
import { CA_PROCESSOR_DATA, type TileProcessorConfig } from 'features/controlLayers/util/controlAdapters';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './ProcessorWrapper';

type Props = ProcessorComponentProps<TileProcessorConfig>;
const DEFAULTS = CA_PROCESSOR_DATA['tile_image_processor'].buildDefaults();

export const TileProcessor = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const tileModeOptions: { label: string; value: TileProcessorMode }[] = useMemo(
    () => [
      { label: t('controlnet.regular'), value: 'regular' },
      { label: t('controlnet.blur'), value: 'blur' },
      { label: t('controlnet.variation'), value: 'var' },
      { label: t('controlnet.super'), value: 'super' },
    ],
    [t]
  );

  const tileModeValue = useMemo(() => tileModeOptions.find((o) => o.value === config.mode), [tileModeOptions, config]);

  const handleTileModeChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isTileProcessorMode(v?.value)) {
        return;
      }
      onChange({ ...config, mode: v.value });
    },
    [config, onChange]
  );

  const handleDownSamplingRateChanged = useCallback(
    (v: number) => {
      onChange({ ...config, down_sampling_rate: v });
    },
    [config, onChange]
  );

  return (
    <ProcessorWrapper>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.mode')}</FormLabel>
        <Combobox value={tileModeValue} options={tileModeOptions} onChange={handleTileModeChange} />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.downsamplingRate')}</FormLabel>
        <CompositeSlider
          value={config.down_sampling_rate}
          onChange={handleDownSamplingRateChanged}
          defaultValue={DEFAULTS.down_sampling_rate}
          min={0}
          max={5}
          step={0.1}
          marks
        />
        <CompositeNumberInput
          value={config.down_sampling_rate}
          onChange={handleDownSamplingRateChanged}
          defaultValue={DEFAULTS.down_sampling_rate}
          min={0}
          max={5}
          step={0.1}
        />
      </FormControl>
    </ProcessorWrapper>
  );
});

TileProcessor.displayName = 'TileProcessor';
