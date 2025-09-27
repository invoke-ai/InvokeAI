import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { bboxScaleMethodChanged } from 'features/controlLayers/store/canvasSlice';
import { selectActiveCanvas } from 'features/controlLayers/store/selectors';
import { isBoundingBoxScaleMethod } from 'features/controlLayers/store/types';
import { useIsBboxSizeLocked } from 'features/parameters/components/Bbox/use-is-bbox-size-locked';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectScaleMethod = createSelector(selectActiveCanvas, (canvas) => canvas.bbox.scaleMethod);

const BboxScaleMethod = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const scaleMethod = useAppSelector(selectScaleMethod);
  const isBboxSizeLocked = useIsBboxSizeLocked();

  const OPTIONS: ComboboxOption[] = useMemo(
    () => [
      { label: t('modelManager.none'), value: 'none' },
      { label: t('common.auto'), value: 'auto' },
      { label: t('modelManager.manual'), value: 'manual' },
    ],
    [t]
  );

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isBoundingBoxScaleMethod(v?.value)) {
        return;
      }
      dispatch(bboxScaleMethodChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => OPTIONS.find((o) => o.value === scaleMethod), [scaleMethod, OPTIONS]);

  return (
    <FormControl isDisabled={isBboxSizeLocked}>
      <InformationalPopover feature="scaleBeforeProcessing">
        <FormLabel>{t('parameters.scaleBeforeProcessing')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(BboxScaleMethod);
