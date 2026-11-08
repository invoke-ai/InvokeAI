import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { bboxAspectRatioIdChanged } from 'features/controlLayers/store/canvasSlice';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import {
  aspectRatioIdChanged,
  selectAllowedAspectRatioIDs,
  selectAspectRatioSizes,
  selectHasFixedDimensionSizes,
} from 'features/controlLayers/store/paramsSlice';
import { selectAspectRatioID } from 'features/controlLayers/store/selectors';
import { isAspectRatioID, zAspectRatioID } from 'features/controlLayers/store/types';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const BboxAspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useAppSelector(selectAspectRatioID);
  const isStaging = useCanvasIsStaging();
  const allowedAspectRatios = useAppSelector(selectAllowedAspectRatioIDs);
  const aspectRatioSizes = useAppSelector(selectAspectRatioSizes);
  const hasFixedSizes = useAppSelector(selectHasFixedDimensionSizes);
  const options = useMemo(() => allowedAspectRatios ?? zAspectRatioID.options, [allowedAspectRatios]);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      if (!isAspectRatioID(e.target.value)) {
        return;
      }
      const fixedSize = aspectRatioSizes?.[e.target.value] ?? undefined;
      dispatch(bboxAspectRatioIdChanged({ id: e.target.value, fixedSize }));
      // For external models with fixed sizes, also sync to params so buildExternalGraph uses correct dimensions
      if (fixedSize) {
        dispatch(aspectRatioIdChanged({ id: e.target.value, fixedSize }));
      }
    },
    [dispatch, aspectRatioSizes]
  );

  return (
    <FormControl isDisabled={isStaging || hasFixedSizes}>
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
