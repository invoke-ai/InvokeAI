import { CompositeRangeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { caLayerBeginEndStepPctChanged, selectCALayer } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  layerId: string;
};

const formatPct = (v: number) => `${Math.round(v * 100)}%`;
const ariaLabel = ['Begin Step %', 'End Step %'];

export const CALayerBeginEndStepPct = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const beginEndStepPct = useAppSelector(
    (s) => selectCALayer(s.controlLayers.present, layerId).controlAdapter.beginEndStepPct
  );

  const onChange = useCallback(
    (v: [number, number]) => {
      dispatch(
        caLayerBeginEndStepPctChanged({
          layerId,
          beginEndStepPct: v,
        })
      );
    },
    [dispatch, layerId]
  );

  const onReset = useCallback(() => {
    dispatch(
      caLayerBeginEndStepPctChanged({
        layerId,
        beginEndStepPct: [0, 1],
      })
    );
  }, [dispatch, layerId]);

  return (
    <FormControl orientation="horizontal">
      <InformationalPopover feature="controlNetBeginEnd">
        <FormLabel m={0}>{t('controlnet.beginEndStepPercentShort')}</FormLabel>
      </InformationalPopover>
      <CompositeRangeSlider
        aria-label={ariaLabel}
        value={beginEndStepPct}
        onChange={onChange}
        onReset={onReset}
        min={0}
        max={1}
        step={0.05}
        fineStep={0.01}
        minStepsBetweenThumbs={1}
        formatValue={formatPct}
        marks
        withThumbTooltip
      />
    </FormControl>
  );
});

CALayerBeginEndStepPct.displayName = 'CALayerBeginEndStepPct';
