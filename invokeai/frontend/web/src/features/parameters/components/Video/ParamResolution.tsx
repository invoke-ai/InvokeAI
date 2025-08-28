import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { aspectRatioIdChanged, heightChanged, widthChanged } from 'features/controlLayers/store/paramsSlice';
import { isVeo3Resolution, VEO3_RESOLUTIONS, zVeo3Resolution } from 'features/controlLayers/store/types';
import { selectVideoResolution, videoResolutionChanged } from 'features/parameters/store/videoSlice';
import type { ChangeEventHandler } from 'react';
import { useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const ParamResolution = () => {
  const videoResolution = useAppSelector(selectVideoResolution);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const options = useMemo(() => zVeo3Resolution.options, []);

  useEffect(() => {
    if (!videoResolution) {
      return;
    }
    dispatch(aspectRatioIdChanged({ id: '16:9' }));
    dispatch(widthChanged({ width: VEO3_RESOLUTIONS[videoResolution].width, updateAspectRatio: true, clamp: true }));
    dispatch(heightChanged({ height: VEO3_RESOLUTIONS[videoResolution].height, updateAspectRatio: true, clamp: true }));
  }, [dispatch, videoResolution]);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      const resolution = e.target.value;
      if (!isVeo3Resolution(resolution)) {
        return;
      }

      dispatch(videoResolutionChanged(resolution));
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
