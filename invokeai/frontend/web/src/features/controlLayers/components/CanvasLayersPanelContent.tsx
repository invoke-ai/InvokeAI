import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion } from 'common/hooks/focus';
import { CanvasAddEntityButtons } from 'features/controlLayers/components/CanvasAddEntityButtons';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityList';
import { EntityListSelectedEntityActionBar } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBar';
import { selectHasEntities } from 'features/controlLayers/store/selectors';
import { memo, useRef } from 'react';

export const CanvasLayersPanelContent = memo(() => {
  const hasEntities = useAppSelector(selectHasEntities);
  const layersPanelFocusRef = useRef<HTMLDivElement>(null);
  useFocusRegion('layers', layersPanelFocusRef);

  return (
    <Flex ref={layersPanelFocusRef} flexDir="column" gap={2} w="full" h="full">
      <EntityListSelectedEntityActionBar />
      <Divider py={0} />
      {!hasEntities && <CanvasAddEntityButtons />}
      {hasEntities && <CanvasEntityList />}
    </Flex>
  );
});

CanvasLayersPanelContent.displayName = 'CanvasLayersPanelContent';
