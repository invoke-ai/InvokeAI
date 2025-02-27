import { MenuItem } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIdentifierBelowThisOne } from 'features/controlLayers/hooks/useNextRenderableEntityIdentifier';
import type { CanvasRenderableEntityType } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStackSimpleBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsMergeDown = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();
  const entityIdentifier = useEntityIdentifierContext<CanvasRenderableEntityType>();
  const entityIdentifierBelowThisOne = useEntityIdentifierBelowThisOne(entityIdentifier);
  const mergeDown = useCallback(() => {
    if (entityIdentifierBelowThisOne === null) {
      return;
    }
    canvasManager.compositor.mergeByEntityIdentifiers([entityIdentifierBelowThisOne, entityIdentifier], true);
  }, [canvasManager.compositor, entityIdentifier, entityIdentifierBelowThisOne]);

  return (
    <MenuItem
      onClick={mergeDown}
      icon={<PiStackSimpleBold />}
      isDisabled={isBusy || entityIdentifierBelowThisOne === null}
    >
      {t('controlLayers.mergeDown')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsMergeDown.displayName = 'CanvasEntityMenuItemsMergeDown';
