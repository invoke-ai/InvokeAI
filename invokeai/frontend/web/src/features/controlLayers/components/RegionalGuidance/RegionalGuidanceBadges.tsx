import { Badge } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const RegionalGuidanceBadges = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const { t } = useTranslation();
  const autoNegative = useAppSelector((s) => selectEntityOrThrow(s.canvasV2.present, entityIdentifier).autoNegative);

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
