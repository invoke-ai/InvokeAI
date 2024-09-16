import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

const createSelectName = (entityIdentifier: CanvasEntityIdentifier) =>
  createSelector(selectCanvasSlice, (canvas) => {
    const entity = selectEntity(canvas, entityIdentifier);
    if (!entity) {
      return null;
    }
    return entity.name;
  });

export const useEntityTitle = (entityIdentifier: CanvasEntityIdentifier) => {
  const { t } = useTranslation();
  const selectName = useMemo(() => createSelectName(entityIdentifier), [entityIdentifier]);
  const name = useAppSelector(selectName);

  const title = useMemo(() => {
    if (name) {
      return name;
    }

    switch (entityIdentifier.type) {
      case 'inpaint_mask':
        return t('controlLayers.inpaintMask');
      case 'control_layer':
        return t('controlLayers.controlLayer');
      case 'raster_layer':
        return t('controlLayers.rasterLayer');
      case 'reference_image':
        return t('controlLayers.globalReferenceImage');
      case 'regional_guidance':
        return t('controlLayers.regionalGuidance');
      default:
        assert(false, 'Unexpected entity type');
    }
  }, [entityIdentifier.type, name, t]);

  return title;
};
