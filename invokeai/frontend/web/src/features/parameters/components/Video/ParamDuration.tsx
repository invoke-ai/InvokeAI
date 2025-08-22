import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isRunwayDurationID,
  isVeo3DurationID,
  RUNWAY_DURATIONS,
  VEO3_DURATIONS,
} from 'features/controlLayers/store/types';
import { selectVideoDuration, selectVideoModel, videoDurationChanged } from 'features/parameters/store/videoSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamDuration = () => {
  const videoDuration = useAppSelector(selectVideoDuration);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const model = useAppSelector(selectVideoModel);

  const options = useMemo(() => {
    if (model?.base === 'veo3') {
      return Object.entries(VEO3_DURATIONS).map(([key, value]) => ({
        label: value,
        value: key,
      }));
    } else if (model?.base === 'runway') {
      return Object.entries(RUNWAY_DURATIONS).map(([key, value]) => ({
        label: value,
        value: key,
      }));
    } else {
      return [];
    }
  }, [model]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      const duration = v?.value;
      if (!isVeo3DurationID(duration) && !isRunwayDurationID(duration)) {
        return;
      }

      dispatch(videoDurationChanged(duration));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === videoDuration), [videoDuration, options]);

  return (
    <FormControl>
      <FormLabel>{t('parameters.duration')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
};
