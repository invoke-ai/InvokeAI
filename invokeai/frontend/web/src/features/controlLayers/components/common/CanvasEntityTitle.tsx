import { Text } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsSelected } from 'features/controlLayers/hooks/useEntityIsSelected';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

export const CanvasEntityTitle = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const isSelected = useEntityIsSelected(entityIdentifier);
  const title = useMemo(() => {
    if (entityIdentifier.type === 'inpaint_mask') {
      return t('controlLayers.inpaintMask');
    } else if (entityIdentifier.type === 'control_adapter') {
      return t('controlLayers.globalControlAdapter');
    } else if (entityIdentifier.type === 'layer') {
      return t('controlLayers.layer');
    } else if (entityIdentifier.type === 'ip_adapter') {
      return t('controlLayers.ipAdapter');
    } else if (entityIdentifier.type === 'regional_guidance') {
      return t('controlLayers.regionalGuidance');
    } else {
      assert(false, 'Unexpected entity type');
    }
  }, [entityIdentifier.type, t]);

  return (
    <Text size="sm" fontWeight="semibold" userSelect="none" color={isSelected ? 'base.100' : 'base.300'}>
      {title}
    </Text>
  );
});

CanvasEntityTitle.displayName = 'CanvasEntityTitle';
