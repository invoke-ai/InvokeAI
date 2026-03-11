import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, dropTargetForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { attachClosestEdge, extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { singleRefImageDndSource } from 'features/dnd/dnd';
import { type DndListTargetState, idle } from 'features/dnd/types';
import { firefoxDndFix } from 'features/dnd/util';
import type { RefObject } from 'react';
import { useEffect, useState } from 'react';

export const useRefImageListDnd = (ref: RefObject<HTMLElement>, id: string) => {
  const [dndListState, setDndListState] = useState<DndListTargetState>(idle);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      firefoxDndFix(element),
      draggable({
        element,
        getInitialData() {
          return singleRefImageDndSource.getData({ id });
        },
        onDragStart() {
          setDndListState({ type: 'is-dragging' });
          setIsDragging(true);
        },
        onDrop() {
          setDndListState(idle);
          setIsDragging(false);
        },
      }),
      dropTargetForElements({
        element,
        canDrop({ source }) {
          if (!singleRefImageDndSource.typeGuard(source.data)) {
            return false;
          }
          return true;
        },
        getData({ input }) {
          const data = singleRefImageDndSource.getData({ id });
          return attachClosestEdge(data, {
            element,
            input,
            allowedEdges: ['left', 'right'],
          });
        },
        getIsSticky() {
          return true;
        },
        onDragEnter({ self }) {
          const closestEdge = extractClosestEdge(self.data);
          setDndListState({ type: 'is-dragging-over', closestEdge });
        },
        onDrag({ self }) {
          const closestEdge = extractClosestEdge(self.data);

          // Only need to update react state if nothing has changed.
          // Prevents re-rendering.
          setDndListState((current) => {
            if (current.type === 'is-dragging-over' && current.closestEdge === closestEdge) {
              return current;
            }
            return { type: 'is-dragging-over', closestEdge };
          });
        },
        onDragLeave() {
          setDndListState(idle);
        },
        onDrop() {
          setDndListState(idle);
        },
      })
    );
  }, [id, ref]);

  return [dndListState, isDragging] as const;
};
