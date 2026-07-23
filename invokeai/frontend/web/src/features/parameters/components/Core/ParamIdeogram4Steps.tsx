import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectIdeogram4SamplerPreset,
  selectIdeogram4Steps,
  setIdeogram4Steps,
} from 'features/controlLayers/store/paramsSlice';
import { IDEOGRAM4_PRESET_DEFAULTS } from 'features/parameters/components/Core/ParamIdeogram4SamplerPreset';
import type React from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

const MARKS = [1, 12, 20, 48, 100];

// Optional override of the sampler preset's step count. null = use the preset.
const ParamIdeogram4Steps = () => {
  const { t } = useTranslation();
  const steps = useAppSelector(selectIdeogram4Steps);
  const preset = useAppSelector(selectIdeogram4SamplerPreset);
  const dispatch = useAppDispatch();

  const presetSteps = IDEOGRAM4_PRESET_DEFAULTS[preset]?.steps ?? 48;
  const onChange = useCallback((v: number) => dispatch(setIdeogram4Steps(v)), [dispatch]);
  const onReset = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dispatch(setIdeogram4Steps(null));
    },
    [dispatch]
  );

  const displayValue = steps ?? presetSteps;

  return (
    <FormControl>
      <FormLabel>
        {t('parameters.steps')}{' '}
        {steps !== null ? (
          <Text as="span" cursor="pointer" onClick={onReset} display="inline-flex" verticalAlign="middle">
            <PiXBold />
          </Text>
        ) : (
          <Text as="span" opacity={0.5} fontWeight="normal" fontSize="xs">
            ({t('common.auto').toLowerCase()})
          </Text>
        )}
      </FormLabel>
      <CompositeSlider
        value={displayValue}
        defaultValue={presetSteps}
        min={1}
        max={100}
        step={1}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={displayValue}
        defaultValue={presetSteps}
        min={1}
        max={100}
        step={1}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamIdeogram4Steps);
