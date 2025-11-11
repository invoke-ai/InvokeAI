import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectCLIPSkip, selectModel, setClipSkip } from 'features/controlLayers/store/paramsSlice';
import { CLIP_SKIP_MAP } from 'features/parameters/types/constants';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 0,
  sliderMin: 0,
  numberInputMin: 0,
  fineStep: 1,
  coarseStep: 1,
};

const ParamClipSkip = () => {
  const clipSkip = useAppSelector(selectCLIPSkip);
  const model = useAppSelector(selectModel);
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
      return CLIP_SKIP_MAP['sd-1']?.maxClip;
    }
    return CLIP_SKIP_MAP[model.base]?.maxClip;
  }, [model]);

  const sliderMarks = useMemo(() => {
    if (!model) {
      return CLIP_SKIP_MAP['sd-1']?.markers;
    }
    return CLIP_SKIP_MAP[model.base]?.markers;
  }, [model]);

  if (model?.base === 'sdxl') {
    return null;
  }

  return (
    <FormControl>
      <InformationalPopover feature="clipSkip">
        <FormLabel>{t('parameters.clipSkip')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={clipSkip}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={max ?? 0}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={handleClipSkipChange}
        marks={sliderMarks}
      />
      <CompositeNumberInput
        value={clipSkip}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.numberInputMin}
        max={max ?? 0}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={handleClipSkipChange}
      />
    </FormControl>
  );
};

export default memo(ParamClipSkip);
