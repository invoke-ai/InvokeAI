import { Badge } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityStateGate } from 'features/controlLayers/contexts/CanvasEntityStateGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const buildSelectWithTransparencyEffect = (entityIdentifier: CanvasEntityIdentifier<'control_layer'>) =>
  createSelector(
    selectCanvasSlice,
    (canvas) => selectEntityOrThrow(canvas, entityIdentifier, 'ControlLayerBadgesContent').withTransparencyEffect
  );

const ControlLayerBadgesContent = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const { t } = useTranslation();
  const selectWithTransparencyEffect = useMemo(
    () => buildSelectWithTransparencyEffect(entityIdentifier),
    [entityIdentifier]
  );
  const withTransparencyEffect = useAppSelector(selectWithTransparencyEffect);

  if (!withTransparencyEffect) {
    return null;
  }

  return (
    <Badge color="base.300" bg="transparent" borderWidth={1} userSelect="none">
      {t('controlLayers.transparency')}
    </Badge>
  );
});

ControlLayerBadgesContent.displayName = 'ControlLayerBadgesContent';

export const ControlLayerBadges = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  return (
    <CanvasEntityStateGate entityIdentifier={entityIdentifier}>
      <ControlLayerBadgesContent />
    </CanvasEntityStateGate>
  );
});
ControlLayerBadges.displayName = 'ControlLayerBadges';
