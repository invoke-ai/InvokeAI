import { Badge } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const ControlLayerBadges = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const { t } = useTranslation();
  const withTransparencyEffect = useAppSelector(
    (s) => selectEntityOrThrow(selectCanvasSlice(s), entityIdentifier).withTransparencyEffect
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
