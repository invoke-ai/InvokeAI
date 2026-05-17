import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPencilSimpleLineBold } from 'react-icons/pi';

export const VectorLayerMenuItemsEdit = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext('vector_layer');
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const editSession = useStore(canvasManager.tool.tools.path.$editSession);

  if (!adapter || adapter.state.type !== 'vector_layer') {
    return null;
  }

  const isEditingThisLayer =
    editSession?.entityIdentifier.id === entityIdentifier.id && editSession.entityIdentifier.type === entityIdentifier.type;

  const onClick = useCallback(() => {
    canvasManager.tool.tools.path.startEdit(entityIdentifier);
  }, [canvasManager.tool.tools.path, entityIdentifier]);

  return (
    <MenuItem
      onClick={onClick}
      icon={<PiPencilSimpleLineBold />}
      isDisabled={adapter.state.paths.length === 0 || isEditingThisLayer}
    >
      {t('common.edit')}
    </MenuItem>
  );
});

VectorLayerMenuItemsEdit.displayName = 'VectorLayerMenuItemsEdit';
