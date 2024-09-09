import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import type { Property } from 'csstype';
import { CanvasHUDItem } from 'features/controlLayers/components/HUD/CanvasHUDItem';
import { useEntityAdapterSafe } from 'features/controlLayers/hooks/useEntityAdapter';
import { useEntityTypeIsHidden } from 'features/controlLayers/hooks/useEntityTypeIsHidden';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import {
  selectCanvasSlice,
  selectEntityOrThrow,
  selectSelectedEntityIdentifier,
} from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { atom } from 'nanostores';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type ContentProps = {
  entityIdentifier: CanvasEntityIdentifier;
  adapter: CanvasEntityAdapter;
};

const $isFilteringFallback = atom(false);

type EntityStatus = {
  value: string;
  color?: Property.Color;
};

const CanvasHUDItemSelectedEntityStatusContent = memo(({ entityIdentifier, adapter }: ContentProps) => {
  const { t } = useTranslation();
  const selectIsEnabled = useMemo(
    () => createSelector(selectCanvasSlice, (canvas) => selectEntityOrThrow(canvas, entityIdentifier).isEnabled),
    [entityIdentifier]
  );
  const selectIsLocked = useMemo(
    () => createSelector(selectCanvasSlice, (canvas) => selectEntityOrThrow(canvas, entityIdentifier).isLocked),
    [entityIdentifier]
  );
  const isEnabled = useAppSelector(selectIsEnabled);
  const isLocked = useAppSelector(selectIsLocked);
  const isHidden = useEntityTypeIsHidden(entityIdentifier.type);
  const isFiltering = useStore(adapter.filterer?.$isFiltering ?? $isFilteringFallback);
  const isTransforming = useStore(adapter.transformer.$isTransforming);

  const status = useMemo<EntityStatus>(() => {
    if (isFiltering) {
      return {
        value: t('controlLayers.HUD.entityStatus.filtering'),
        color: 'invokeYellow.300',
      };
    }

    if (isTransforming) {
      return {
        value: t('controlLayers.HUD.entityStatus.transforming'),
        color: 'invokeYellow.300',
      };
    }

    if (isHidden) {
      return {
        value: t('controlLayers.HUD.entityStatus.hidden'),
        color: 'invokePurple.300',
      };
    }

    if (isLocked) {
      return {
        value: t('controlLayers.HUD.entityStatus.locked'),
        color: 'invokeRed.300',
      };
    }

    if (!isEnabled) {
      return {
        value: t('controlLayers.HUD.entityStatus.disabled'),
        color: 'invokeRed.300',
      };
    }

    return {
      value: t('controlLayers.HUD.entityStatus.enabled'),
    };
  }, [isFiltering, isTransforming, isHidden, isLocked, isEnabled, t]);

  return (
    <>
      <CanvasHUDItem
        label={t('controlLayers.HUD.entityStatus.selectedEntity')}
        value={status.value}
        color={status.color}
      />
    </>
  );
});

CanvasHUDItemSelectedEntityStatusContent.displayName = 'CanvasHUDItemSelectedEntityStatusContent';

export const CanvasHUDItemSelectedEntityStatus = memo(() => {
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const adapter = useEntityAdapterSafe(selectedEntityIdentifier);

  if (!selectedEntityIdentifier || !adapter) {
    return null;
  }

  return <CanvasHUDItemSelectedEntityStatusContent entityIdentifier={selectedEntityIdentifier} adapter={adapter} />;
});

CanvasHUDItemSelectedEntityStatus.displayName = 'CanvasHUDItemSelectedEntityStatus';
