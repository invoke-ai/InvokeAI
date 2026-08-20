import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { EntityListSelectedEntityActionBarFill } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarFill';
import { EntityListSelectedEntityActionBarOpacity } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBarOpacity';
import { VectorLayerTraceWidth } from 'features/controlLayers/components/VectorLayer/VectorLayerTraceWidth';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

import { EntityListSelectedEntityActionBarCompositeOperation } from './EntityListSelectedEntityActionBarCompositeOperation';

export const EntityListSelectedEntityActionBar = memo(() => {
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isVectorLayerSelected = selectedEntityIdentifier?.type === 'vector_layer';

  return (
    <Flex flexDirection="column" gap={2}>
      <Flex w="full" minW={0} gap={2} ps={1}>
        <EntityListSelectedEntityActionBarCompositeOperation />
        <EntityListSelectedEntityActionBarOpacity />
        {isVectorLayerSelected && <VectorLayerTraceWidth />}
        <EntityListSelectedEntityActionBarFill />
      </Flex>
    </Flex>
  );
});

EntityListSelectedEntityActionBar.displayName = 'EntityListSelectedEntityActionBar';
