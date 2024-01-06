import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvRangeSlider } from 'common/components/InvRangeSlider/InvRangeSlider';
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

  const value = useMemo<[number, number]>(
    () => [stepPcts?.beginStepPct ?? 0, stepPcts?.endStepPct ?? 1],
    [stepPcts]
  );

  if (!stepPcts) {
    return null;
  }

  return (
    <InvControl
      isDisabled={!isEnabled}
      label={t('controlnet.beginEndStepPercent')}
      feature="controlNetBeginEnd"
      orientation="vertical"
    >
      <InvRangeSlider
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
    </InvControl>
  );
});

ParamControlAdapterBeginEnd.displayName = 'ParamControlAdapterBeginEnd';

const ariaLabel = ['Begin Step %', 'End Step %'];
