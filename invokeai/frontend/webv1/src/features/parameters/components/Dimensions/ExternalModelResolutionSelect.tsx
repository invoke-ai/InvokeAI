import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  aspectRatioIdChanged,
  selectAspectRatioID,
  selectAspectRatioSizes,
} from 'features/controlLayers/store/paramsSlice';
import { isAspectRatioID } from 'features/controlLayers/store/types';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const ExternalModelResolutionSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const aspectRatioID = useAppSelector(selectAspectRatioID);
  const aspectRatioSizes = useAppSelector(selectAspectRatioSizes);

  const options = useMemo(() => {
    if (!aspectRatioSizes) {
      return [];
    }
    return Object.entries(aspectRatioSizes).map(([ratio, size]) => ({
      ratio,
      label: `${ratio} (${size.width}×${size.height})`,
      size,
    }));
  }, [aspectRatioSizes]);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      const ratio = e.target.value;
      if (!isAspectRatioID(ratio)) {
        return;
      }
      const fixedSize = aspectRatioSizes?.[ratio] ?? undefined;
      dispatch(aspectRatioIdChanged({ id: ratio, fixedSize }));
    },
    [dispatch, aspectRatioSizes]
  );

  if (!aspectRatioSizes) {
    return null;
  }

  return (
    <FormControl>
      <FormLabel>{t('parameters.resolution')}</FormLabel>
      <Select
        size="sm"
        value={aspectRatioID}
        onChange={onChange}
        cursor="pointer"
        iconSize="0.75rem"
        icon={<PiCaretDownBold />}
      >
        {options.map(({ ratio, label }) => (
          <option key={ratio} value={ratio}>
            {label}
          </option>
        ))}
      </Select>
    </FormControl>
  );
});

ExternalModelResolutionSelect.displayName = 'ExternalModelResolutionSelect';
