import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAddEntityButtons } from 'features/controlLayers/components/CanvasAddEntityButtons';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityList';
import { EntityListActionBar } from 'features/controlLayers/components/CanvasEntityList/EntityListActionBar';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectHasEntities } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

export const CanvasPanelContent = memo(() => {
  const hasEntities = useAppSelector(selectHasEntities);

  return (
    <CanvasManagerProviderGate>
      <Flex flexDir="column" gap={2} w="full" h="full">
        <EntityListActionBar />
        <Divider py={0} />
        {!hasEntities && <CanvasAddEntityButtons />}
        {hasEntities && <CanvasEntityList />}
      </Flex>
    </CanvasManagerProviderGate>
  );
});

CanvasPanelContent.displayName = 'CanvasPanelContent';
