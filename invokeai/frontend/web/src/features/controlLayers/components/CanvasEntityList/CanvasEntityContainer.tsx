import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCanvasEntityListDnd } from 'features/controlLayers/components/CanvasEntityList/useCanvasEntityListDnd';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsSelected } from 'features/controlLayers/hooks/useEntityIsSelected';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import { DndListDropIndicator } from 'features/dnd/DndListDropIndicator';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useRef } from 'react';

const sx = {
  position: 'relative',
  flexDir: 'column',
  w: 'full',
  bg: 'base.850',
  borderRadius: 'base',
  '&[data-selected=true]': {
    bg: 'base.800',
  },
  '&[data-is-dragging=true]': {
    opacity: 0.3,
  },
  transitionProperty: 'common',
} satisfies SystemStyleObject;

export const CanvasEntityContainer = memo((props: PropsWithChildren) => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const isSelected = useEntityIsSelected(entityIdentifier);
  const onClick = useCallback(() => {
    if (isSelected) {
      return;
    }
    dispatch(entitySelected({ entityIdentifier }));
  }, [dispatch, entityIdentifier, isSelected]);
  const ref = useRef<HTMLDivElement>(null);

  const [dndListState, isDragging] = useCanvasEntityListDnd(ref, entityIdentifier);

  return (
    <Box position="relative">
      <Flex
        // This is used to trigger the post-move flash animation
        data-entity-id={entityIdentifier.id}
        data-selected={isSelected}
        data-is-dragging={isDragging}
        ref={ref}
        onClick={onClick}
        sx={sx}
      >
        {props.children}
      </Flex>
      <DndListDropIndicator dndState={dndListState} />
    </Box>
  );
});

CanvasEntityContainer.displayName = 'CanvasEntityContainer';
