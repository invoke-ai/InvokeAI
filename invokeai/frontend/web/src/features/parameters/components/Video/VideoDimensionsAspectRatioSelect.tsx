import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { isVideoAspectRatio, zRunwayAspectRatioID, zVeo3AspectRatioID } from 'features/controlLayers/store/types';
import {
  selectIsRunway,
  selectIsVeo3,
  selectVideoAspectRatio,
  videoAspectRatioChanged,
} from 'features/parameters/store/videoSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const VideoDimensionsAspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useAppSelector(selectVideoAspectRatio);
  const isVeo3 = useAppSelector(selectIsVeo3);
  const isRunway = useAppSelector(selectIsRunway);
  const options = useMemo(() => {
    if (isVeo3) {
      return zVeo3AspectRatioID.options.map((o) => ({ label: o, value: o }));
    }
    if (isRunway) {
      return zRunwayAspectRatioID.options.map((o) => ({ label: o, value: o }));
    }
    // All other models
    return [];
  }, [isVeo3, isRunway]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isVideoAspectRatio(v?.value)) {
        return;
      }
      dispatch(videoAspectRatioChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === id), [id, options]);

  return (
    <FormControl>
      <InformationalPopover feature="paramAspect">
        <FormLabel>{t('parameters.aspect')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
});

VideoDimensionsAspectRatioSelect.displayName = 'VideoDimensionsAspectRatioSelect';
