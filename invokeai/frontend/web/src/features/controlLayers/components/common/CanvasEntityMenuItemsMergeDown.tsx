import { MenuItem } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { getIsCanvasMergeDownSupported } from 'features/controlLayers/hooks/canvasMergeHotkeyUtils';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIdentifierBelowThisOne } from 'features/controlLayers/hooks/useNextRenderableEntityIdentifier';
import type { CanvasEntityType } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStackSimpleBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsMergeDown = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();
  const entityIdentifier = useEntityIdentifierContext<CanvasEntityType>();
  const entityIdentifierBelowThisOne = useEntityIdentifierBelowThisOne(entityIdentifier);
  const isSupported = getIsCanvasMergeDownSupported(entityIdentifier, entityIdentifierBelowThisOne);
  const mergeDown = useCallback(() => {
    if (entityIdentifierBelowThisOne === null || !isSupported) {
      return;
    }
    void canvasManager.compositor.mergeDown(entityIdentifierBelowThisOne, entityIdentifier);
  }, [canvasManager.compositor, entityIdentifier, entityIdentifierBelowThisOne, isSupported]);

  return (
    <MenuItem
      onClick={mergeDown}
      icon={<PiStackSimpleBold />}
      isDisabled={isBusy || entityIdentifierBelowThisOne === null || !isSupported}
    >
      {t('controlLayers.mergeDown')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsMergeDown.displayName = 'CanvasEntityMenuItemsMergeDown';
