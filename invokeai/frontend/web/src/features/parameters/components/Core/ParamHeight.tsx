import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { documentHeightChanged } from 'features/controlLayers/store/canvasV2Slice';
import { selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamHeight = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const height = useAppSelector((s) => s.canvasV2.document.rect.height);
  const sliderMin = useAppSelector((s) => s.config.sd.height.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.height.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.height.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.height.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.height.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.height.fineStep);

  const onChange = useCallback(
    (v: number) => {
      dispatch(documentHeightChanged({ height: v }));
    },
    [dispatch]
  );

  const marks = useMemo(() => [sliderMin, optimalDimension, sliderMax], [sliderMin, optimalDimension, sliderMax]);

  return (
    <FormControl>
      <InformationalPopover feature="paramHeight">
        <FormLabel>{t('parameters.height')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={height}
        defaultValue={optimalDimension}
        onChange={onChange}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        marks={marks}
      />
      <CompositeNumberInput
        value={height}
        defaultValue={optimalDimension}
        onChange={onChange}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
      />
    </FormControl>
  );
});

ParamHeight.displayName = 'ParamHeight';
