import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, dropTargetForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { attachClosestEdge, extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import type { DndListTargetState } from 'features/dnd/types';
import { idle } from 'features/dnd/types';
import type { ActionData, ActionSourceApi } from 'features/imageActions/actions';
import { buildGetData, buildTypeAndKey, buildTypeGuard } from 'features/imageActions/actions';
import type { FieldIdentifier } from 'features/nodes/types/field';
import type { RefObject } from 'react';
import { useEffect, useState } from 'react';

const _singleWorkflowField = buildTypeAndKey('single-workflow-field');
type SingleWorkflowFieldSourceData = ActionData<
  typeof _singleWorkflowField.type,
  typeof _singleWorkflowField.key,
  { fieldIdentifier: FieldIdentifier }
>;
export const singleWorkflowField: ActionSourceApi<SingleWorkflowFieldSourceData> = {
  ..._singleWorkflowField,
  typeGuard: buildTypeGuard(_singleWorkflowField.key),
  getData: buildGetData(_singleWorkflowField.key, _singleWorkflowField.type),
};

export const useLinearViewFieldDnd = (ref: RefObject<HTMLElement>, fieldIdentifier: FieldIdentifier) => {
  const [dndListState, setListDndState] = useState<DndListTargetState>(idle);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      draggable({
        element,
        getInitialData() {
          return singleWorkflowField.getData({ fieldIdentifier });
        },
        onDragStart() {
          setListDndState({ type: 'is-dragging' });
          setIsDragging(true);
        },
        onDrop() {
          setListDndState(idle);
          setIsDragging(false);
        },
      }),
      dropTargetForElements({
        element,
        canDrop({ source }) {
          if (!singleWorkflowField.typeGuard(source.data)) {
            return false;
          }
          return true;
        },
        getData({ input }) {
          const data = singleWorkflowField.getData({ fieldIdentifier });
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
          setListDndState({ type: 'is-dragging-over', closestEdge });
        },
        onDrag({ self }) {
          const closestEdge = extractClosestEdge(self.data);

          // Only need to update react state if nothing has changed.
          // Prevents re-rendering.
          setListDndState((current) => {
            if (current.type === 'is-dragging-over' && current.closestEdge === closestEdge) {
              return current;
            }
            return { type: 'is-dragging-over', closestEdge };
          });
        },
        onDragLeave() {
          setListDndState(idle);
        },
        onDrop() {
          setListDndState(idle);
        },
      })
    );
  }, [fieldIdentifier, ref]);

  return [dndListState, isDragging] as const;
};
