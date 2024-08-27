import { Box, ContextMenu, MenuList } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAddEntityButtons } from 'features/controlLayers/components/CanvasAddEntityButtons';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityList';
import { CanvasEntityListMenuItems } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityListMenuItems';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectHasEntities } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';

export const CanvasPanelContent = memo(() => {
  const hasEntities = useAppSelector(selectHasEntities);
  const renderMenu = useCallback(
    () => (
      <MenuList>
        <CanvasEntityListMenuItems />
      </MenuList>
    ),
    []
  );
  return (
    <CanvasManagerProviderGate>
      <ContextMenu<HTMLDivElement> renderMenu={renderMenu}>
        {(ref) => (
          <Box ref={ref} w="full" h="full">
            {!hasEntities && <CanvasAddEntityButtons />}
            {hasEntities && <CanvasEntityList />}
          </Box>
        )}
      </ContextMenu>
    </CanvasManagerProviderGate>
  );
});

CanvasPanelContent.displayName = 'CanvasPanelContent';
