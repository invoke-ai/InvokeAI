import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isRunwayDurationID,
  isVeo3DurationID,
  RUNWAY_DURATIONS,
  VEO3_DURATIONS,
} from 'features/controlLayers/store/types';
import { selectVideoDuration, selectVideoModel, videoDurationChanged } from 'features/parameters/store/videoSlice';
import type { ChangeEventHandler } from 'react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

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

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      const duration = e.target.value;
      if (!isVeo3DurationID(duration) && !isRunwayDurationID(duration)) {
        return;
      }

      dispatch(videoDurationChanged(duration));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === videoDuration)?.value, [videoDuration, options]);

  return (
    <FormControl>
      <FormLabel>{t('parameters.duration')}</FormLabel>
      <Select
        size="sm"
        value={value}
        onChange={onChange}
        cursor="pointer"
        iconSize="0.75rem"
        icon={<PiCaretDownBold />}
      >
        {options.map((duration) => (
          <option key={duration.value} value={duration.value}>
            {duration.label}
          </option>
        ))}
      </Select>
    </FormControl>
  );
};
