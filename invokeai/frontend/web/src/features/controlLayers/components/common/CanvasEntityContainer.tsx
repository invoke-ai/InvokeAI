import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, dropTargetForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { attachClosestEdge, extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsSelected } from 'features/controlLayers/hooks/useEntityIsSelected';
import { useEntitySelectionColor } from 'features/controlLayers/hooks/useEntitySelectionColor';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import type { DndState } from 'features/dnd/dnd';
import { Dnd, idle } from 'features/dnd/dnd';
import { DndDropIndicator } from 'features/dnd/DndDropIndicator';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';

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
  const [dndState, setDndState] = useState<DndState>(idle);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      draggable({
        element,
        getInitialData() {
          return Dnd.Source.singleCanvasEntity.getData({ entityIdentifier });
        },
        onDragStart() {
          setDndState({ type: 'is-dragging' });
        },
        onDrop() {
          setDndState(idle);
        },
      }),
      dropTargetForElements({
        element,
        canDrop({ source }) {
          if (!Dnd.Source.singleCanvasEntity.typeGuard(source.data)) {
            return false;
          }
          if (source.data.payload.entityIdentifier.type !== entityIdentifier.type) {
            return false;
          }
          return true;
        },
        getData({ input }) {
          const data = Dnd.Source.singleCanvasEntity.getData({ entityIdentifier });
          return attachClosestEdge(data, {
            element,
            input,
            allowedEdges: ['top', 'bottom'],
          });
        },
        getIsSticky() {
          return true;
        },
        onDragEnter({ self }) {
          const closestEdge = extractClosestEdge(self.data);
          setDndState({ type: 'is-dragging-over', closestEdge });
        },
        onDrag({ self }) {
          const closestEdge = extractClosestEdge(self.data);

          // Only need to update react state if nothing has changed.
          // Prevents re-rendering.
          setDndState((current) => {
            if (current.type === 'is-dragging-over' && current.closestEdge === closestEdge) {
              return current;
            }
            return { type: 'is-dragging-over', closestEdge };
          });
        },
        onDragLeave() {
          setDndState(idle);
        },
        onDrop() {
          setDndState(idle);
        },
      })
    );
  }, [entityIdentifier]);

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
      {dndState.type === 'is-dragging-over' && dndState.closestEdge ? (
        <DndDropIndicator
          edge={dndState.closestEdge}
          // This is the gap between items in the list
          gap="var(--invoke-space-2)"
        />
      ) : null}
    </Box>
  );
});

CanvasEntityContainer.displayName = 'CanvasEntityContainer';
