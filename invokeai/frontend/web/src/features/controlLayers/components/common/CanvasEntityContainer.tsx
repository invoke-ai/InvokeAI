import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, dropTargetForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
// import { pointerOutsideOfPreview } from '@atlaskit/pragmatic-drag-and-drop/element/pointer-outside-of-preview';
// import { setCustomNativeDragPreview } from '@atlaskit/pragmatic-drag-and-drop/element/set-custom-native-drag-preview';
import type { Edge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { attachClosestEdge, extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { Flex } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsSelected } from 'features/controlLayers/hooks/useEntityIsSelected';
import { useEntitySelectionColor } from 'features/controlLayers/hooks/useEntitySelectionColor';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import { Dnd } from 'features/dnd/dnd';
import DropIndicator from 'features/dnd/DndDropIndicator';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';

type DndState =
  | {
      type: 'idle';
    }
  | {
      type: 'preview';
      container: HTMLElement;
    }
  | {
      type: 'is-dragging';
    }
  | {
      type: 'is-dragging-over';
      closestEdge: Edge | null;
    };

const idle: DndState = { type: 'idle' };

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
        // onGenerateDragPreview({ nativeSetDragImage }) {
        //   setCustomNativeDragPreview({
        //     nativeSetDragImage,
        //     getOffset: pointerOutsideOfPreview({
        //       x: '16px',
        //       y: '8px',
        //     }),
        //     render({ container }) {
        //       setState({ type: 'preview', container });
        //     },
        //   });
        // },
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
          // not allowing dropping on yourself
          if (source.element === element) {
            return false;
          }
          // only allowing tasks to be dropped on me
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
    <Flex
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
      {dndState.type === 'is-dragging-over' && dndState.closestEdge ? (
        <DropIndicator edge={dndState.closestEdge} gap="8px" />
      ) : null}
    </Flex>
  );
});

CanvasEntityContainer.displayName = 'CanvasEntityContainer';
