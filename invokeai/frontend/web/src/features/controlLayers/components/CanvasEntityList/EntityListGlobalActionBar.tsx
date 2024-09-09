import { Flex, Spacer } from '@invoke-ai/ui-library';
import { EntityListGlobalActionBarAddLayerMenu } from 'features/controlLayers/components/CanvasEntityList/EntityListGlobalActionBarAddLayerMenu';
import { EntityListGlobalActionBarDenoisingStrength } from 'features/controlLayers/components/CanvasEntityList/EntityListGlobalActionBarDenoisingStrength';
import { CanvasToolbarFitBboxToLayersButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarFitBboxToLayersButton';
import { memo } from 'react';

export const EntityListGlobalActionBar = memo(() => {
  return (
    <Flex w="full" gap={2} alignItems="center">
      <EntityListGlobalActionBarDenoisingStrength />
      <Spacer />
      <Flex>
        <CanvasToolbarFitBboxToLayersButton />
        <EntityListGlobalActionBarAddLayerMenu />
      </Flex>
    </Flex>
  );
});

EntityListGlobalActionBar.displayName = 'EntityListGlobalActionBar';
