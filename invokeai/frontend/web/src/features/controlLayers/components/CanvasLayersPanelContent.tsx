import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { CanvasAddEntityButtons } from 'features/controlLayers/components/CanvasAddEntityButtons';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityList';
import { EntityListSelectedEntityActionBar } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBar';
import { selectHasEntities } from 'features/controlLayers/store/selectors';
import { memo, useRef } from 'react';

export const CanvasLayersPanelContent = memo(() => {
  const hasEntities = useAppSelector(selectHasEntities);
  const layersPanelFocusRef = useRef<HTMLDivElement>(null);
  useFocusRegion('layers', layersPanelFocusRef);
  const isRegionFocused = useIsRegionFocused('layers');

  return (
    <Flex
      ref={layersPanelFocusRef}
      flexDir="column"
      gap={2}
      w="full"
      h="full"
      borderWidth={1}
      borderColor={isRegionFocused ? 'blue.300' : 'transparent'}
      borderRadius="base"
      p={2}
      marginTop={-1}
      transition="border-color 0.1s"
    >
      <EntityListSelectedEntityActionBar />
      <Divider py={0} borderColor="base.600" />
      {!hasEntities && <CanvasAddEntityButtons />}
      {hasEntities && <CanvasEntityList />}
    </Flex>
  );
});

CanvasLayersPanelContent.displayName = 'CanvasLayersPanelContent';
