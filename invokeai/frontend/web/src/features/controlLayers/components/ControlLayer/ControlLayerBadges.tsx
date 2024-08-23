import { Badge } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectControlLayerEntityOrThrow } from 'features/controlLayers/store/controlLayersReducers';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const ControlLayerBadges = memo(() => {
  const { id } = useEntityIdentifierContext();
  const { t } = useTranslation();
  const withTransparencyEffect = useAppSelector(
    (s) => selectControlLayerEntityOrThrow(s.canvasV2, id).withTransparencyEffect
  );

  return (
    <>
      {withTransparencyEffect && (
        <Badge color="base.300" bg="transparent" borderWidth={1} userSelect="none">
          {t('controlLayers.transparency')}
        </Badge>
      )}
    </>
  );
});

ControlLayerBadges.displayName = 'ControlLayerBadges';
