import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setClipSkip } from 'features/parameters/store/generationSlice';
import { CLIP_SKIP_MAP } from 'features/parameters/types/constants';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamClipSkip = () => {
  const clipSkip = useAppSelector((s) => s.generation.clipSkip);
  const initial = useAppSelector((s) => s.config.sd.clipSkip.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.clipSkip.sliderMin);
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.clipSkip.numberInputMin
  );
  const coarseStep = useAppSelector((s) => s.config.sd.clipSkip.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.clipSkip.fineStep);
  const { model } = useAppSelector((s) => s.generation);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleClipSkipChange = useCallback(
    (v: number) => {
      dispatch(setClipSkip(v));
    },
    [dispatch]
  );

  const max = useMemo(() => {
    if (!model) {
      return CLIP_SKIP_MAP['sd-1'].maxClip;
    }
    return CLIP_SKIP_MAP[model.base_model].maxClip;
  }, [model]);

  const sliderMarks = useMemo(() => {
    if (!model) {
      return CLIP_SKIP_MAP['sd-1'].markers;
    }
    return CLIP_SKIP_MAP[model.base_model].markers;
  }, [model]);

  if (model?.base_model === 'sdxl') {
    return null;
  }

  return (
    <InvControl label={t('parameters.clipSkip')} feature="clipSkip">
      <InvSlider
        value={clipSkip}
        defaultValue={initial}
        min={sliderMin}
        max={max}
        step={coarseStep}
        fineStep={fineStep}
        onChange={handleClipSkipChange}
        withNumberInput
        numberInputMin={numberInputMin}
        numberInputMax={max}
        marks={sliderMarks}
      />
    </InvControl>
  );
};

export default memo(ParamClipSkip);
