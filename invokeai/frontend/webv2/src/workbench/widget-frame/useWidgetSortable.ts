import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent } from 'react';

import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { getWidgetInstanceDragData, getWidgetInstanceDragId } from '@workbench/widgetDnd';
import { useCanEditStructure } from '@workbench/workspace/useStructureLock';
import { useCallback } from 'react';

const WIDGET_FILL_TRANSITION =
  'background var(--chakra-durations-faster) ease, border-color var(--chakra-durations-faster) ease, color var(--chakra-durations-faster) ease';

const NO_DRAG_PROPS: Record<string, unknown> = {};

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
  // Locked, a tab is not sortable: no listeners and no sortable announcement, but it stays a focusable control.
  const isLocked = !useCanEditStructure();
  const { attributes, isDragging, listeners, setActivatorNodeRef, setNodeRef, transform, transition } = useSortable({
    data: getWidgetInstanceDragData(region, instanceId, typeId),
    disabled: disabled || isLocked,
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
      // Ark Tabs handles Space/Enter before user bubble handlers. Start the sortable keyboard gesture on capture so
      // Space remains available for reorder, while Enter keeps its native tab-activation behavior.
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
    dragHandleProps: isLocked
      ? { role: attributes.role, tabIndex: attributes.tabIndex }
      : { ...attributes, ...listeners },
    isDragging,
    // Preserve host roles, state, and roving tab indices while adding dnd listeners/descriptions.
    semanticDragHandleProps: isLocked
      ? NO_DRAG_PROPS
      : {
          ...semanticAttributes,
          ...semanticListeners,
          onKeyDownCapture: handleSemanticKeyDownCapture,
        },
    setNodeRef: setSortableNodeRef,
    style: {
      transform: CSS.Transform.toString(transform),
      // Compose fill fading with dnd-kit's inline transform transition.
      transition: [transition, WIDGET_FILL_TRANSITION].filter(Boolean).join(', '),
    },
  };
};
