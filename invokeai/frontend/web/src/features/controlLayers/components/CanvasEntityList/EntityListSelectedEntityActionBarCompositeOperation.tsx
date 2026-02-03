import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rasterLayerGlobalCompositeOperationChanged } from 'features/controlLayers/store/canvasSlice';
import {
  selectCanvasSlice,
  selectEntity,
  selectSelectedEntityIdentifier,
} from 'features/controlLayers/store/selectors';
import type {
  CanvasEntityIdentifier,
  CanvasRasterLayerState,
  CompositeOperation,
} from 'features/controlLayers/store/types';
import { COLOR_BLEND_MODES } from 'features/controlLayers/store/types';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectCompositeOperation = createSelector(selectCanvasSlice, (canvas) => {
  const { selectedEntityIdentifier } = canvas;

  if (selectedEntityIdentifier?.type !== 'raster_layer') {
    return 'source-over';
  }

  const entity = selectEntity(canvas, selectedEntityIdentifier);

  return (entity as CanvasRasterLayerState)?.globalCompositeOperation ?? 'source-over';
});

export const EntityListSelectedEntityActionBarCompositeOperation = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const currentOperation = useAppSelector(selectCompositeOperation);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      if (selectedEntityIdentifier?.type === 'raster_layer') {
        const value = e.target.value as CompositeOperation;

        dispatch(
          rasterLayerGlobalCompositeOperationChanged({
            entityIdentifier: selectedEntityIdentifier as CanvasEntityIdentifier<'raster_layer'>,
            globalCompositeOperation: value,
          })
        );
      }
    },
    [dispatch, selectedEntityIdentifier]
  );

  if (selectedEntityIdentifier?.type !== 'raster_layer') {
    return null;
  }

  return (
    <FormControl w="min-content" gap={2}>
      <FormLabel m={0} mt={1} whiteSpace="nowrap">
        {t('controlLayers.compositeOperation.label')}
      </FormLabel>
      <Select value={currentOperation} onChange={onChange} size="sm" variant="outline" minW="110px">
        {COLOR_BLEND_MODES.map((op) => (
          <option key={op} value={op}>
            {t(`controlLayers.compositeOperation.blendModes.${op}`)}
          </option>
        ))}
      </Select>
    </FormControl>
  );
});

EntityListSelectedEntityActionBarCompositeOperation.displayName = 'EntityListSelectedEntityActionBarCompositeOperation';
