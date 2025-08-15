import { FormControl, FormLabel, Select } from "@invoke-ai/ui-library";
import { useAppDispatch, useAppSelector } from "app/store/storeHooks";
import { selectVideoDuration, setVideoDuration } from "features/controlLayers/store/paramsSlice";
import { isParameterDuration, ParameterDuration } from "features/parameters/types/parameterSchemas";
import { ChangeEventHandler, useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { PiCaretDownBold } from "react-icons/pi";

const options: { label: string; value: ParameterDuration }[] = [
    { label: '5 seconds', value: 5 },
    { label: '10 seconds', value: 10 },
  ];

export const ParamDuration = () => {
  const videoDuration = useAppSelector(selectVideoDuration);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      if (!isParameterDuration(e.target.value)) {
        return;
      }

      dispatch(setVideoDuration(e.target.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === videoDuration), [videoDuration]);

  return <FormControl>
    <FormLabel>{t('parameters.duration')}</FormLabel>
    <Select size="sm" value={value?.value} onChange={onChange} cursor="pointer" iconSize="0.75rem" icon={<PiCaretDownBold />}>
        {options.map((duration) => (
          <option key={duration.value} value={duration.value}>
            {duration.label}
          </option>
        ))}
      </Select>
  </FormControl>;
};