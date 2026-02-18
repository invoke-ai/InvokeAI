import { Flex } from '@invoke-ai/ui-library';
import { EntityListGlobalActionBarAddLayerMenu } from 'features/controlLayers/components/CanvasEntityList/EntityListGlobalActionBarAddLayerMenu';
import { EntityListSelectedEntityActionBarDuplicateButton } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarDuplicateButton';
import { EntityListSelectedEntityActionBarFilterButton } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarFilterButton';
import { EntityListSelectedEntityActionBarInvertMaskButton } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarInvertMaskButton';
import { EntityListSelectedEntityActionBarSelectObjectButton } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarSelectObjectButton';
import { EntityListSelectedEntityActionBarTransformButton } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarTransformButton';
import { EntityListNonRasterLayerToggle } from 'features/controlLayers/components/common/CanvasNonRasterLayersIsHiddenToggle';
import { memo } from 'react';

import { EntityListSelectedEntityActionBarSaveToAssetsButton } from './EntityListSelectedEntityActionBarSaveToAssetsButton';

export const EntityListSelectedEntityOperationsBar = memo(() => {
  return (
    <Flex w="full" minH="20px" gap={1} alignItems="center" ps={2} pr={2}>
      <EntityListGlobalActionBarAddLayerMenu />
      <EntityListSelectedEntityActionBarTransformButton />
      <EntityListSelectedEntityActionBarDuplicateButton />
      <EntityListSelectedEntityActionBarSelectObjectButton />
      <EntityListSelectedEntityActionBarFilterButton />
      <EntityListSelectedEntityActionBarInvertMaskButton />
      <EntityListNonRasterLayerToggle />
      <EntityListSelectedEntityActionBarSaveToAssetsButton />
    </Flex>
  );
});

EntityListSelectedEntityOperationsBar.displayName = 'EntityListSelectedEntityOperationsBar';
