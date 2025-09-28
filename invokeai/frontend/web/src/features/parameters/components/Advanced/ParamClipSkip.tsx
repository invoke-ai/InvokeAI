import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectCLIPSkip, selectModel, setClipSkip, useParamsDispatch } from 'features/controlLayers/store/paramsSlice';
import { CLIP_SKIP_MAP } from 'features/parameters/types/constants';
import { selectCLIPSkipConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamClipSkip = () => {
  const clipSkip = useAppSelector(selectCLIPSkip);
  const config = useAppSelector(selectCLIPSkipConfig);
  const model = useAppSelector(selectModel);

  const dispatchParams = useParamsDispatch();
  const { t } = useTranslation();

  const handleClipSkipChange = useCallback(
    (v: number) => {
      dispatchParams(setClipSkip, v);
    },
    [dispatchParams]
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
        defaultValue={config.initial}
        min={config.sliderMin}
        max={max ?? 0}
        step={config.coarseStep}
        fineStep={config.fineStep}
        onChange={handleClipSkipChange}
        marks={sliderMarks}
      />
      <CompositeNumberInput
        value={clipSkip}
        defaultValue={config.initial}
        min={config.numberInputMin}
        max={max ?? 0}
        step={config.coarseStep}
        fineStep={config.fineStep}
        onChange={handleClipSkipChange}
      />
    </FormControl>
  );
};

export default memo(ParamClipSkip);
