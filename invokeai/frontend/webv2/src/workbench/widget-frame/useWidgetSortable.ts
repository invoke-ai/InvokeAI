import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent } from 'react';

import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { getWidgetInstanceDragData, getWidgetInstanceDragId } from '@workbench/widgetDnd';
import { useCallback } from 'react';

export const useWidgetSortable = ({
  disabled,
  instanceId,
  region,
  typeId,
}: {
  region: WidgetRegion;
  instanceId: WidgetInstanceId;
  typeId: WidgetTypeId;
  disabled?: boolean;
}): {
  setNodeRef: ReturnType<typeof useSortable>['setNodeRef'];
  style: CSSProperties;
  dragHandleProps: Record<string, unknown>;
  semanticDragHandleProps: Record<string, unknown>;
  isDragging: boolean;
} => {
  const { attributes, isDragging, listeners, setActivatorNodeRef, setNodeRef, transform, transition } = useSortable({
    data: getWidgetInstanceDragData(region, instanceId, typeId),
    disabled,
    id: getWidgetInstanceDragId(region, instanceId),
  });
  const setSortableNodeRef = useCallback(
    (node: HTMLElement | null) => {
      setNodeRef(node);
      setActivatorNodeRef(node);
    },
    [setActivatorNodeRef, setNodeRef]
  );
  const handleSemanticKeyDownCapture = useCallback(
    (event: ReactKeyboardEvent<HTMLElement>) => {
      // Ark Tabs handles Space/Enter before user bubble handlers. Start the
      // sortable keyboard gesture on capture so Space remains available for
      // reorder, while Enter keeps its native tab-activation behavior.
      if (event.code === 'Space') {
        listeners?.onKeyDown?.(event);
      }
    },
    [listeners]
  );
  const {
    'aria-disabled': _ariaDisabled,
    'aria-pressed': _ariaPressed,
    'aria-roledescription': _ariaRoleDescription,
    role: _role,
    tabIndex: _tabIndex,
    ...semanticAttributes
  } = attributes;
  const { onKeyDown: _onKeyDown, ...semanticListeners } = listeners ?? {};

  return {
    dragHandleProps: { ...attributes, ...listeners },
    isDragging,
    // Tabs and other semantic controls already own their role, state, and
    // roving tab index. Keep dnd-kit's pointer/keyboard listeners and
    // description without letting its generic "button" contract overwrite
    // the host control's accessibility semantics.
    semanticDragHandleProps: {
      ...semanticAttributes,
      ...semanticListeners,
      onKeyDownCapture: handleSemanticKeyDownCapture,
    },
    setNodeRef: setSortableNodeRef,
    style: { transform: CSS.Transform.toString(transform), transition },
  };
};
