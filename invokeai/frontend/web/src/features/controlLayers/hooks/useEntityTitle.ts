import { useEntityObjectCount } from 'features/controlLayers/hooks/useEntityObjectCount';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

export const useEntityTitle = (entityIdentifier: CanvasEntityIdentifier) => {
  const { t } = useTranslation();

  const objectCount = useEntityObjectCount(entityIdentifier);

  const title = useMemo(() => {
    const parts: string[] = [];
    if (entityIdentifier.type === 'inpaint_mask') {
      parts.push(t('controlLayers.inpaintMask'));
    } else if (entityIdentifier.type === 'control_adapter') {
      parts.push(t('controlLayers.globalControlAdapter'));
    } else if (entityIdentifier.type === 'layer') {
      parts.push(t('controlLayers.layer'));
    } else if (entityIdentifier.type === 'ip_adapter') {
      parts.push(t('controlLayers.ipAdapter'));
    } else if (entityIdentifier.type === 'regional_guidance') {
      parts.push(t('controlLayers.regionalGuidance'));
    } else {
      assert(false, 'Unexpected entity type');
    }

    if (objectCount > 0) {
      parts.push(`(${objectCount})`);
    }

    return parts.join(' ');
  }, [entityIdentifier.type, objectCount, t]);

  return title;
};
