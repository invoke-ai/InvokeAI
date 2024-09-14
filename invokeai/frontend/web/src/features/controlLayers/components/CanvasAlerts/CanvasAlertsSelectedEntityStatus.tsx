import type { AlertStatus } from '@invoke-ai/ui-library';
import { Alert, AlertIcon, AlertTitle } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityTitle } from 'features/controlLayers/hooks/useEntityTitle';
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

type AlertData = {
  status: AlertStatus;
  title: string;
};

const CanvasAlertsSelectedEntityStatusContent = memo(({ entityIdentifier, adapter }: ContentProps) => {
  const { t } = useTranslation();
  const title = useEntityTitle(entityIdentifier);
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
  const isEmpty = useStore(adapter.$isEmpty);

  const alert = useMemo<AlertData | null>(() => {
    if (isFiltering) {
      return {
        status: 'info',
        title: t('controlLayers.HUD.entityStatus.isFiltering', { title }),
      };
    }

    if (isTransforming) {
      return {
        status: 'info',
        title: t('controlLayers.HUD.entityStatus.isTransforming', { title }),
      };
    }

    if (isEmpty) {
      return {
        status: 'info',
        title: t('controlLayers.HUD.entityStatus.isEmpty', { title }),
      };
    }

    if (isHidden) {
      return {
        status: 'warning',
        title: t('controlLayers.HUD.entityStatus.isHidden', { title }),
      };
    }

    if (isLocked) {
      return {
        status: 'warning',
        title: t('controlLayers.HUD.entityStatus.isLocked', { title }),
      };
    }

    if (!isEnabled) {
      return {
        status: 'warning',
        title: t('controlLayers.HUD.entityStatus.isDisabled', { title }),
      };
    }

    return null;
  }, [isFiltering, isTransforming, isEmpty, isHidden, isLocked, isEnabled, title, t]);

  if (!alert) {
    return null;
  }

  return (
    <Alert status={alert.status} borderRadius="base" fontSize="sm" shadow="md" w="fit-content" alignSelf="flex-end">
      <AlertIcon />
      <AlertTitle>{alert.title}</AlertTitle>
    </Alert>
  );
});

CanvasAlertsSelectedEntityStatusContent.displayName = 'CanvasAlertsSelectedEntityStatusContent';

export const CanvasAlertsSelectedEntityStatus = memo(() => {
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const adapter = useEntityAdapterSafe(selectedEntityIdentifier);

  if (!selectedEntityIdentifier || !adapter) {
    return null;
  }

  return <CanvasAlertsSelectedEntityStatusContent entityIdentifier={selectedEntityIdentifier} adapter={adapter} />;
});

CanvasAlertsSelectedEntityStatus.displayName = 'CanvasAlertsSelectedEntityStatus';
