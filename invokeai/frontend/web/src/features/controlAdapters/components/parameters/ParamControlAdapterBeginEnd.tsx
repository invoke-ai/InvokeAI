import { CompositeRangeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useControlAdapterBeginEndStepPct } from 'features/controlAdapters/hooks/useControlAdapterBeginEndStepPct';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import {
  controlAdapterBeginStepPctChanged,
  controlAdapterEndStepPctChanged,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

const formatPct = (v: number) => `${Math.round(v * 100)}%`;

export const ParamControlAdapterBeginEnd = memo(({ id }: Props) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const stepPcts = useControlAdapterBeginEndStepPct(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: [number, number]) => {
      dispatch(
        controlAdapterBeginStepPctChanged({
          id,
          beginStepPct: v[0],
        })
      );
      dispatch(
        controlAdapterEndStepPctChanged({
          id,
          endStepPct: v[1],
        })
      );
    },
    [dispatch, id]
  );

  const onReset = useCallback(() => {
    dispatch(
      controlAdapterBeginStepPctChanged({
        id,
        beginStepPct: 0,
      })
    );
    dispatch(
      controlAdapterEndStepPctChanged({
        id,
        endStepPct: 1,
      })
    );
  }, [dispatch, id]);

  const value = useMemo<[number, number]>(() => [stepPcts?.beginStepPct ?? 0, stepPcts?.endStepPct ?? 1], [stepPcts]);

  if (!stepPcts) {
    return null;
  }

  return (
    <FormControl isDisabled={!isEnabled} orientation="vertical">
      <InformationalPopover feature="controlNetBeginEnd">
        <FormLabel>{t('controlnet.beginEndStepPercent')}</FormLabel>
      </InformationalPopover>
      <CompositeRangeSlider
        aria-label={ariaLabel}
        value={value}
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

ParamControlAdapterBeginEnd.displayName = 'ParamControlAdapterBeginEnd';

const ariaLabel = ['Begin Step %', 'End Step %'];
