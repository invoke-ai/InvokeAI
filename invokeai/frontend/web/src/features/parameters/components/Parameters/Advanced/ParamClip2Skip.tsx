import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
import { setClip2Skip } from 'features/parameters/store/generationSlice';
import { clipSkipMap } from 'features/parameters/types/constants';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  stateSelector,
  ({ generation }) => {
    const { model, clip2Skip } = generation;
    return { model, clip2Skip };
  },
  defaultSelectorOptions
);

const ParamClip2Skip = () => {
  const { model, clip2Skip } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleClipSkipChange = useCallback(
    (v: number) => {
      dispatch(setClip2Skip(v));
    },
    [dispatch]
  );

  const handleClipSkipReset = useCallback(() => {
    dispatch(setClip2Skip(0));
  }, [dispatch]);

  const max = useMemo(() => {
    if (!model) {
      return clipSkipMap['sd-1'].maxClip;
    }
    return clipSkipMap[model.base_model].maxClip;
  }, [model]);

  const sliderMarks = useMemo(() => {
    if (!model) {
      return clipSkipMap['sd-1'].markers;
    }
    return clipSkipMap[model.base_model].markers;
  }, [model]);

  return (
    <IAIInformationalPopover details="clipSkip">
      <IAISlider
        label={t('parameters.clip2Skip')}
        aria-label={t('parameters.clip2Skip')}
        min={0}
        max={max}
        step={1}
        value={clip2Skip}
        onChange={handleClipSkipChange}
        withSliderMarks
        sliderMarks={sliderMarks}
        withInput
        withReset
        handleReset={handleClipSkipReset}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamClip2Skip);
