import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityObjectCount } from 'features/controlLayers/hooks/useEntityObjectCount';
import { selectCanvasV2Slice, selectEntity } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

const createSelectName = (entityIdentifier: CanvasEntityIdentifier) =>
  createSelector(selectCanvasV2Slice, (canvasV2) => {
    const entity = selectEntity(canvasV2, entityIdentifier);
    if (!entity) {
      return null;
    }
    return entity.name;
  });

export const useEntityTitle = (entityIdentifier: CanvasEntityIdentifier) => {
  const { t } = useTranslation();
  const selectName = useMemo(() => createSelectName(entityIdentifier), [entityIdentifier]);
  const name = useAppSelector(selectName);
  const objectCount = useEntityObjectCount(entityIdentifier);

  const title = useMemo(() => {
    if (name) {
      return name;
    }

    const parts: string[] = [];
    if (entityIdentifier.type === 'inpaint_mask') {
      parts.push(t('controlLayers.inpaintMask', { count: 1 }));
    } else if (entityIdentifier.type === 'control_layer') {
      parts.push(t('controlLayers.controlLayer', { count: 1 }));
    } else if (entityIdentifier.type === 'raster_layer') {
      parts.push(t('controlLayers.rasterLayer', { count: 1 }));
    } else if (entityIdentifier.type === 'ip_adapter') {
      parts.push(t('common.ipAdapter', { count: 1 }));
    } else if (entityIdentifier.type === 'regional_guidance') {
      parts.push(t('controlLayers.regionalGuidance', { count: 1 }));
    } else {
      assert(false, 'Unexpected entity type');
    }

    if (objectCount > 0) {
      parts.push(`(${objectCount})`);
    }

    return parts.join(' ');
  }, [entityIdentifier.type, name, objectCount, t]);

  return title;
};
