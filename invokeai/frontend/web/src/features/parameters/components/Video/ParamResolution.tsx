import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { heightChanged, widthChanged } from 'features/controlLayers/store/paramsSlice';
import {
  isVeo3Resolution,
  VEO3_RESOLUTIONS,
  zRunwayResolution,
  zVeo3Resolution,
} from 'features/controlLayers/store/types';
import { selectVideoModel, selectVideoResolution, videoResolutionChanged } from 'features/parameters/store/videoSlice';
import type { ChangeEventHandler } from 'react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const ParamResolution = () => {
  const videoResolution = useAppSelector(selectVideoResolution);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const model = useAppSelector(selectVideoModel);

  const options = useMemo(() => {
    if (model?.base === 'veo3') {
      return zVeo3Resolution.options;
    } else if (model?.base === 'runway') {
      return zRunwayResolution.options;
    } else {
      return [];
    }
  }, [model]);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      const resolution = e.target.value;
      if (!isVeo3Resolution(resolution)) {
        return;
      }

      dispatch(videoResolutionChanged(resolution));
      dispatch(widthChanged({ width: VEO3_RESOLUTIONS[resolution].width, updateAspectRatio: true, clamp: true }));
      dispatch(heightChanged({ height: VEO3_RESOLUTIONS[resolution].height, updateAspectRatio: true, clamp: true }));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o === videoResolution), [videoResolution, options]);

  return (
    <FormControl>
      <FormLabel>{t('parameters.resolution')}</FormLabel>
      <Select
        size="sm"
        value={value}
        onChange={onChange}
        cursor="pointer"
        iconSize="0.75rem"
        icon={<PiCaretDownBold />}
      >
        {options.map((resolution) => (
          <option key={resolution} value={resolution}>
            {resolution}
          </option>
        ))}
      </Select>
    </FormControl>
  );
};
