import { Flex } from '@invoke-ai/ui-library';
import { EntityListSelectedEntityActionBarFill } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarFill';
import { EntityListSelectedEntityActionBarOpacity } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarOpacity';
import { memo } from 'react';

import { EntityListSelectedEntityActionBarCompositeOperation } from './EntityListSelectedEntityActionBarCompositeOperation';

export const EntityListSelectedEntityActionBar = memo(() => {
  return (
    <Flex flexDirection="column" gap={2}>
      <Flex w="full" gap={2} ps={1}>
        <EntityListSelectedEntityActionBarCompositeOperation />
        <EntityListSelectedEntityActionBarOpacity />
        <EntityListSelectedEntityActionBarFill />
      </Flex>
    </Flex>
  );
});

EntityListSelectedEntityActionBar.displayName = 'EntityListSelectedEntityActionBar';
