import { FormControl, FormLabel, Select } from "@invoke-ai/ui-library";
import { useAppDispatch, useAppSelector } from "app/store/storeHooks";
import {  selectVideoDuration, videoDurationChanged } from "features/parameters/store/videoSlice";
import { isVeo3DurationID, VEO3_DURATIONS, zVeo3DurationID } from "features/controlLayers/store/types";
import type { ChangeEventHandler} from "react";
import { useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { PiCaretDownBold } from "react-icons/pi";

export const ParamDuration = () => {
  const videoDuration = useAppSelector(selectVideoDuration);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const options = useMemo(() => {
 
      return Object.entries(VEO3_DURATIONS).map(([key, value]) => ({
        label: value,
        value: key,
      }));
   
  }, []);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      const duration = e.target.value;
      if (!isVeo3DurationID(duration)) {
        return;
      }

      dispatch(videoDurationChanged(duration));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === videoDuration)?.value, [videoDuration]);

  return <FormControl>
    <FormLabel>{t('parameters.duration')}</FormLabel>
    <Select size="sm" value={value} onChange={onChange} cursor="pointer" iconSize="0.75rem" icon={<PiCaretDownBold />}>
        {options.map((duration) => (
          <option key={duration.value} value={duration.value}>
            {duration.label}
          </option>
        ))}
      </Select>
  </FormControl>;
};