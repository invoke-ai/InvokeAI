import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectIdeogram4Mu,
  selectIdeogram4SamplerPreset,
  setIdeogram4Mu,
} from 'features/controlLayers/store/paramsSlice';
import { IDEOGRAM4_PRESET_DEFAULTS } from 'features/parameters/components/Core/ParamIdeogram4SamplerPreset';
import type React from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

const MARKS = [0, 0.5, 1, 1.5, 2];

// Optional override of the logit-normal schedule mean (mu). null = use the preset's mu.
const ParamIdeogram4Mu = () => {
  const { t } = useTranslation();
  const mu = useAppSelector(selectIdeogram4Mu);
  const preset = useAppSelector(selectIdeogram4SamplerPreset);
  const dispatch = useAppDispatch();

  const presetMu = IDEOGRAM4_PRESET_DEFAULTS[preset]?.mu ?? 0;
  const onChange = useCallback((v: number) => dispatch(setIdeogram4Mu(v)), [dispatch]);
  const onReset = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dispatch(setIdeogram4Mu(null));
    },
    [dispatch]
  );

  const displayValue = mu ?? presetMu;

  return (
    <FormControl>
      <FormLabel>
        {t('parameters.ideogram4ScheduleShift')}{' '}
        {mu !== null ? (
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
        defaultValue={presetMu}
        min={0}
        max={2}
        step={0.05}
        fineStep={0.01}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={displayValue}
        defaultValue={presetMu}
        min={-4}
        max={4}
        step={0.05}
        fineStep={0.01}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamIdeogram4Mu);
