import { Flex, Spacer } from '@invoke-ai/ui-library';
import { EntityListGlobalActionBarAddLayerMenu } from 'features/controlLayers/components/CanvasEntityList/EntityListGlobalActionBarAddLayerMenu';
import { EntityListGlobalActionBarDenoisingStrength } from 'features/controlLayers/components/CanvasEntityList/EntityListGlobalActionBarDenoisingStrength';
import { memo } from 'react';

export const EntityListGlobalActionBar = memo(() => {
  return (
    <Flex w="full" py={1} px={1} gap={2} alignItems="center">
      <EntityListGlobalActionBarDenoisingStrength />
      <Spacer />
      <EntityListGlobalActionBarAddLayerMenu />
    </Flex>
  );
});

EntityListGlobalActionBar.displayName = 'EntityListGlobalActionBar';
