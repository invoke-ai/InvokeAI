import { Badge } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const RegionalGuidanceBadges = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const { t } = useTranslation();
  const selectAutoNegative = useMemo(
    () => createSelector(selectCanvasSlice, (canvas) => selectEntityOrThrow(canvas, entityIdentifier).autoNegative),
    [entityIdentifier]
  );
  const autoNegative = useAppSelector(selectAutoNegative);

  return (
    <>
      {autoNegative && (
        <Badge color="base.300" bg="transparent" borderWidth={1} userSelect="none">
          {t('controlLayers.autoNegative')}
        </Badge>
      )}
    </>
  );
});

RegionalGuidanceBadges.displayName = 'RegionalGuidanceBadges';
