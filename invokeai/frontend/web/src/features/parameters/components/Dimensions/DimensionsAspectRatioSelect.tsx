import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { aspectRatioIdChanged, selectAspectRatioID } from 'features/controlLayers/store/paramsSlice';
import { isAspectRatioID, zAspectRatioID } from 'features/controlLayers/store/types';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const DimensionsAspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useAppSelector(selectAspectRatioID);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      if (!isAspectRatioID(e.target.value)) {
        return;
      }
      dispatch(aspectRatioIdChanged({ id: e.target.value }));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramAspect">
        <FormLabel>{t('parameters.aspect')}</FormLabel>
      </InformationalPopover>
      <Select size="sm" value={id} onChange={onChange} cursor="pointer" iconSize="0.75rem" icon={<PiCaretDownBold />}>
        {zAspectRatioID.options.map((ratio) => (
          <option key={ratio} value={ratio}>
            {ratio}
          </option>
        ))}
      </Select>
    </FormControl>
  );
});

DimensionsAspectRatioSelect.displayName = 'DimensionsAspectRatioSelect';
