import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { bboxAspectRatioIdChanged } from 'features/controlLayers/store/canvasSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectIsImagen3 } from 'features/controlLayers/store/paramsSlice';
import { selectAspectRatioID } from 'features/controlLayers/store/selectors';
import { isAspectRatioID, zAspectRatioID, zImagen3AspectRatioID } from 'features/controlLayers/store/types';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const BboxAspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useAppSelector(selectAspectRatioID);
  const isStaging = useAppSelector(selectIsStaging);
  const isImagen3 = useAppSelector(selectIsImagen3);

  const options = useMemo(() => {
    if (!isImagen3) {
      return zAspectRatioID.options;
    }
    return zImagen3AspectRatioID.options;
  }, [isImagen3]);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      if (!isAspectRatioID(e.target.value)) {
        return;
      }
      dispatch(bboxAspectRatioIdChanged({ id: e.target.value }));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={isStaging}>
      <InformationalPopover feature="paramAspect">
        <FormLabel>{t('parameters.aspect')}</FormLabel>
      </InformationalPopover>
      <Select size="sm" value={id} onChange={onChange} cursor="pointer" iconSize="0.75rem" icon={<PiCaretDownBold />}>
        {options.map((ratio) => (
          <option key={ratio} value={ratio}>
            {ratio}
          </option>
        ))}
      </Select>
    </FormControl>
  );
});

BboxAspectRatioSelect.displayName = 'BboxAspectRatioSelect';
