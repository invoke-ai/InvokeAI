import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  minimaxH3DurationSecondsChanged,
  selectMiniMaxH3DurationSeconds,
  selectMiniMaxH3OutputMode,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

// MiniMax H3 runs at a fixed 24 fps and frame counts snap to the 17n+5 grid. The slider works
// in whole seconds; the graph builder snaps to the nearest legal frame count, and the 14 s
// stop maps to the model's true ceiling (345 frames = 14.375 s).
const CONSTRAINTS = {
  initial: 5,
  sliderMin: 5,
  sliderMax: 14,
  fineStep: 1,
  coarseStep: 1,
};

const MARKS = [5, 10, 14];

/**
 * MiniMax H3 video duration in seconds. Only shown in the 'video' output mode - the 'image'
 * mode always generates the 5-frame minimum block.
 */
const ParamMiniMaxH3Duration = () => {
  const { t } = useTranslation();
  const duration = useAppSelector(selectMiniMaxH3DurationSeconds);
  const outputMode = useAppSelector(selectMiniMaxH3OutputMode);
  const dispatch = useAppDispatch();

  const onChange = useCallback((v: number) => dispatch(minimaxH3DurationSecondsChanged(v)), [dispatch]);

  if (outputMode !== 'video') {
    return null;
  }

  return (
    <FormControl>
      <FormLabel>{t('parameters.minimaxH3DurationSeconds')}</FormLabel>
      <CompositeSlider
        value={duration}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={duration}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamMiniMaxH3Duration);
