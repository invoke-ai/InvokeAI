import type { ComboboxOption, SystemStyleObject } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { SingleValue } from 'chakra-react-select';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { bboxAspectRatioIdChanged } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { ASPECT_RATIO_OPTIONS } from 'features/parameters/components/DocumentSize/constants';
import { isAspectRatioID } from 'features/parameters/components/DocumentSize/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectAspectRatioID = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.aspectRatio.id);

export const AspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useAppSelector(selectAspectRatioID);

  const onChange = useCallback(
    (v: SingleValue<ComboboxOption>) => {
      if (!v || !isAspectRatioID(v.value)) {
        return;
      }
      dispatch(bboxAspectRatioIdChanged({ id: v.value }));
    },
    [dispatch]
  );

  const value = useMemo(() => ASPECT_RATIO_OPTIONS.filter((o) => o.value === id)[0], [id]);

  return (
    <FormControl>
      <InformationalPopover feature="paramAspect">
        <FormLabel>{t('parameters.aspect')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} onChange={onChange} options={ASPECT_RATIO_OPTIONS} sx={selectStyles} />
    </FormControl>
  );
});

AspectRatioSelect.displayName = 'AspectRatioSelect';

const selectStyles: SystemStyleObject = { minW: 24 };
