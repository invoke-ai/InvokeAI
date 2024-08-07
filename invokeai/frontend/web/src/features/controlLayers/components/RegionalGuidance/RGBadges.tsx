import { Badge } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectRGOrThrow } from 'features/controlLayers/store/regionsReducers';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const RGBadges = memo(() => {
  const { id } = useEntityIdentifierContext();
  const { t } = useTranslation();
  const autoNegative = useAppSelector((s) => selectRGOrThrow(s.canvasV2, id).autoNegative);

  return (
    <>
      {autoNegative === 'invert' && (
        <Badge color="base.300" bg="transparent" borderWidth={1} userSelect="none">
          {t('controlLayers.autoNegative')}
        </Badge>
      )}
    </>
  );
});

RGBadges.displayName = 'RGBadges';
