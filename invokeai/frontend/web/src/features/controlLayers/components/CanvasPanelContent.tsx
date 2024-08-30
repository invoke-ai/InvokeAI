import { Box, ContextMenu, Divider, Flex, MenuList } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAddEntityButtons } from 'features/controlLayers/components/CanvasAddEntityButtons';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityList';
import { EntityListActionBar } from 'features/controlLayers/components/CanvasEntityList/EntityListActionBar';
import { CanvasEntityListMenuItems } from 'features/controlLayers/components/CanvasEntityList/EntityListActionBarAddLayerMenuItems';
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
      <Flex flexDir="column" gap={2} w="full" h="full">
        <EntityListActionBar />
        <Divider py={0} />
        <ContextMenu<HTMLDivElement> renderMenu={renderMenu}>
          {(ref) => (
            <Box ref={ref} w="full" h="full">
              {!hasEntities && <CanvasAddEntityButtons />}
              {hasEntities && <CanvasEntityList />}
            </Box>
          )}
        </ContextMenu>
      </Flex>
    </CanvasManagerProviderGate>
  );
});

CanvasPanelContent.displayName = 'CanvasPanelContent';
