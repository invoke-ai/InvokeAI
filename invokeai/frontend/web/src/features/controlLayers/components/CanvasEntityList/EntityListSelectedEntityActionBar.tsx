import { Flex, Spacer } from '@invoke-ai/ui-library';
import { EntityListSelectedEntityActionBarDeleteButton } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarDeleteButton';
import { EntityListSelectedEntityActionBarDuplicateButton } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarDuplicateButton';
import { EntityListSelectedEntityActionBarFill } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarFill';
import { EntityListSelectedEntityActionBarFilterButton } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarFilterButton';
import { EntityListSelectedEntityActionBarOpacity } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarOpacity';
import { EntityListSelectedEntityActionBarTransformButton } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarTransformButton';
import { memo } from 'react';

export const EntityListSelectedEntityActionBar = memo(() => {
  return (
    <Flex w="full" py={1} px={1} gap={2} alignItems="center">
      <EntityListSelectedEntityActionBarOpacity />
      <Spacer />
      <EntityListSelectedEntityActionBarFill />
      <Flex>
        <EntityListSelectedEntityActionBarFilterButton />
        <EntityListSelectedEntityActionBarTransformButton />
        <EntityListSelectedEntityActionBarDuplicateButton />
        <EntityListSelectedEntityActionBarDeleteButton />
      </Flex>
    </Flex>
  );
});

EntityListSelectedEntityActionBar.displayName = 'EntityListSelectedEntityActionBar';
