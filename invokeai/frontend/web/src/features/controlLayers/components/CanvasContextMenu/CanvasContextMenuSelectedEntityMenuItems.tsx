import { MenuGroup } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityMenuItemsCopy } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCopy';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsFilter } from 'features/controlLayers/components/common/CanvasEntityMenuItemsFilter';
import { CanvasEntityMenuItemsSave } from 'features/controlLayers/components/common/CanvasEntityMenuItemsSave';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import {
  EntityIdentifierContext,
  useEntityIdentifierContext,
} from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityTitle } from 'features/controlLayers/hooks/useEntityTitle';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isFilterableEntityIdentifier, isTransformableEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';

const CanvasContextMenuSelectedEntityMenuItemsContent = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  const title = useEntityTitle(entityIdentifier);

  return (
    <MenuGroup title={title}>
      {isFilterableEntityIdentifier(entityIdentifier) && <CanvasEntityMenuItemsFilter />}
      {isTransformableEntityIdentifier(entityIdentifier) && <CanvasEntityMenuItemsTransform />}
      <CanvasEntityMenuItemsCopy />
      <CanvasEntityMenuItemsSave />
      <CanvasEntityMenuItemsDelete />
    </MenuGroup>
  );
});
CanvasContextMenuSelectedEntityMenuItemsContent.displayName = 'CanvasContextMenuSelectedEntityMenuItemsContent';

export const CanvasContextMenuSelectedEntityMenuItems = memo(() => {
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);

  if (!selectedEntityIdentifier) {
    return null;
  }

  return (
    <EntityIdentifierContext.Provider value={selectedEntityIdentifier}>
      <CanvasContextMenuSelectedEntityMenuItemsContent />
    </EntityIdentifierContext.Provider>
  );
});

CanvasContextMenuSelectedEntityMenuItems.displayName = 'CanvasContextMenuSelectedEntityMenuItems';
