import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAddEntityButtons } from 'features/controlLayers/components/CanvasAddEntityButtons';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityList';
import { EntityListGlobalActionBar } from 'features/controlLayers/components/CanvasEntityList/EntityListGlobalActionBar';
import { EntityListSelectedEntityActionBar } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBar';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectHasEntities } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

export const CanvasPanelContent = memo(() => {
  const hasEntities = useAppSelector(selectHasEntities);

  return (
    <CanvasManagerProviderGate>
      <Flex flexDir="column" gap={2} w="full" h="full">
        <EntityListGlobalActionBar />
        <Divider py={0} />
        <EntityListSelectedEntityActionBar />
        <Divider py={0} />
        {!hasEntities && <CanvasAddEntityButtons />}
        {hasEntities && <CanvasEntityList />}
      </Flex>
    </CanvasManagerProviderGate>
  );
});

CanvasPanelContent.displayName = 'CanvasPanelContent';
