import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectCLIPSKip, selectModel, setClipSkip } from 'features/controlLayers/store/paramsSlice';
import { CLIP_SKIP_MAP } from 'features/parameters/types/constants';
import { selectCLIPSkipConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamClipSkip = () => {
  const clipSkip = useAppSelector(selectCLIPSKip);
  const config = useAppSelector(selectCLIPSkipConfig);
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
      return CLIP_SKIP_MAP['sd-1'].maxClip;
    }
    return CLIP_SKIP_MAP[model.base].maxClip;
  }, [model]);

  const sliderMarks = useMemo(() => {
    if (!model) {
      return CLIP_SKIP_MAP['sd-1'].markers;
    }
    return CLIP_SKIP_MAP[model.base].markers;
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
        defaultValue={config.initial}
        min={config.sliderMin}
        max={max}
        step={config.coarseStep}
        fineStep={config.fineStep}
        onChange={handleClipSkipChange}
        marks={sliderMarks}
      />
      <CompositeNumberInput
        value={clipSkip}
        defaultValue={config.initial}
        min={config.numberInputMin}
        max={max}
        step={config.coarseStep}
        fineStep={config.fineStep}
        onChange={handleClipSkipChange}
      />
    </FormControl>
  );
};

export default memo(ParamClipSkip);
