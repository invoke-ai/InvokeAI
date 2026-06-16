import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectWanGuidanceScaleLowNoise,
  wanGuidanceScaleLowNoiseChanged,
} from 'features/controlLayers/store/paramsSlice';
import type React from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

// Match the primary ParamCFGScale's range so the slider thumb position is
// visually comparable between the two CFG sliders at the same numeric value
// (e.g. CFG=5 and CFG-Low=3 should look correct relative to each other).
const CONSTRAINTS = {
  initial: 3.5,
  sliderMin: 1,
  sliderMax: 20,
  numberInputMin: 1,
  numberInputMax: 200,
  fineStep: 0.1,
  coarseStep: 0.5,
};

const MARKS = [CONSTRAINTS.sliderMin, Math.floor(CONSTRAINTS.sliderMax / 2), CONSTRAINTS.sliderMax];

/**
 * Wan 2.2 Guidance Scale (Low Noise)
 *
 * Optional separate CFG for the A14B low-noise expert. When null (cleared),
 * the denoise node falls back to the primary guidance_scale. Ignored for
 * TI2V-5B (single-expert).
 *
 * Diffusers reference defaults for A14B: primary 4.0 / low-noise 3.0 — i.e.
 * a slightly lower CFG on the detail-pass expert produces less over-sharpened
 * output.
 */
const ParamWanGuidanceScaleLowNoise = () => {
  const { t } = useTranslation();
  const value = useAppSelector(selectWanGuidanceScaleLowNoise);
  const dispatch = useAppDispatch();

  const onChange = useCallback((v: number) => dispatch(wanGuidanceScaleLowNoiseChanged(v)), [dispatch]);
  const onReset = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dispatch(wanGuidanceScaleLowNoiseChanged(null));
    },
    [dispatch]
  );

  const displayValue = value ?? CONSTRAINTS.initial;

  return (
    <FormControl>
      <FormLabel>
        {t('parameters.wanGuidanceScaleLowNoise')}{' '}
        {value !== null && (
          <IconButton
            size="xs"
            variant="link"
            aria-label={t('common.reset')}
            icon={<PiXBold />}
            onClick={onReset}
            minW={4}
            h={4}
          />
        )}
      </FormLabel>
      <CompositeSlider
        value={displayValue}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={displayValue}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamWanGuidanceScaleLowNoise);
