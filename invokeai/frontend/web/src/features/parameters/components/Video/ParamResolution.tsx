import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isRunwayResolution,
  isVeo3Resolution,
  zRunwayResolution,
  zVeo3Resolution,
} from 'features/controlLayers/store/types';
import { selectVideoModel, selectVideoResolution, videoResolutionChanged } from 'features/parameters/store/videoSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamResolution = () => {
  const videoResolution = useAppSelector(selectVideoResolution);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const model = useAppSelector(selectVideoModel);

  const options = useMemo(() => {
    if (model?.base === 'veo3') {
      return zVeo3Resolution.options.map((o) => ({ label: o, value: o }));
    } else if (model?.base === 'runway') {
      return zRunwayResolution.options.map((o) => ({ label: o, value: o }));
    } else {
      return [];
    }
  }, [model]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      const resolution = v?.value;
      if (!isVeo3Resolution(resolution) && !isRunwayResolution(resolution)) {
        return;
      }

      dispatch(videoResolutionChanged(resolution));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === videoResolution), [videoResolution, options]);

  return (
    <FormControl>
      <FormLabel>{t('parameters.resolution')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
};
