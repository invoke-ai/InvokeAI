import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCanvasEntityListDnd } from 'features/controlLayers/components/CanvasEntityList/useCanvasEntityListDnd';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsSelected } from 'features/controlLayers/hooks/useEntityIsSelected';
import { useEntitySelectionColor } from 'features/controlLayers/hooks/useEntitySelectionColor';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import { DndListDropIndicator } from 'features/dnd/DndListDropIndicator';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useRef } from 'react';

export const CanvasEntityContainer = memo((props: PropsWithChildren) => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const isSelected = useEntityIsSelected(entityIdentifier);
  const selectionColor = useEntitySelectionColor(entityIdentifier);
  const onClick = useCallback(() => {
    if (isSelected) {
      return;
    }
    dispatch(entitySelected({ entityIdentifier }));
  }, [dispatch, entityIdentifier, isSelected]);
  const ref = useRef<HTMLDivElement>(null);

  const dndState = useCanvasEntityListDnd(ref, entityIdentifier);

  return (
    <Box position="relative">
      <Flex
        // This is used to trigger the post-move flash animation
        data-entity-id={entityIdentifier.id}
        ref={ref}
        position="relative"
        flexDir="column"
        w="full"
        bg={isSelected ? 'base.800' : 'base.850'}
        onClick={onClick}
        borderInlineStartWidth={5}
        borderColor={isSelected ? selectionColor : 'base.800'}
        borderRadius="base"
      >
        {props.children}
      </Flex>
      <DndListDropIndicator dndState={dndState} />
    </Box>
  );
});

CanvasEntityContainer.displayName = 'CanvasEntityContainer';
