import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityTypeIsHidden } from 'features/controlLayers/hooks/useEntityTypeIsHidden';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPencilSimpleLineBold } from 'react-icons/pi';

export const VectorLayerMenuItemsEdit = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext('vector_layer');
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const isVectorLayerTypeHidden = useEntityTypeIsHidden('vector_layer');
  const editSession = useStore(canvasManager.tool.tools.path.$editSession);
  const onClick = useCallback(() => {
    canvasManager.tool.tools.path.startEdit(entityIdentifier);
  }, [canvasManager.tool.tools.path, entityIdentifier]);

  if (!adapter || adapter.state.type !== 'vector_layer') {
    return null;
  }

  const isEditingThisLayer =
    editSession?.entityIdentifier.id === entityIdentifier.id &&
    editSession.entityIdentifier.type === entityIdentifier.type;

  return (
    <MenuItem
      onClick={onClick}
      icon={<PiPencilSimpleLineBold />}
      isDisabled={
        adapter.state.paths.length === 0 ||
        !adapter.state.isEnabled ||
        adapter.state.isLocked ||
        isVectorLayerTypeHidden ||
        isBusy ||
        isEditingThisLayer
      }
    >
      {t('common.edit')}
    </MenuItem>
  );
});

VectorLayerMenuItemsEdit.displayName = 'VectorLayerMenuItemsEdit';
