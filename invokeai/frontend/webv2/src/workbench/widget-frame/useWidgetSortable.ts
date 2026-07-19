import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';
import type { CSSProperties } from 'react';

import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { getWidgetInstanceDragData, getWidgetInstanceDragId } from '@workbench/widgetDnd';

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
  isDragging: boolean;
} => {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({
    data: getWidgetInstanceDragData(region, instanceId, typeId),
    disabled,
    id: getWidgetInstanceDragId(region, instanceId),
  });

  return {
    dragHandleProps: { ...attributes, ...listeners },
    isDragging,
    setNodeRef,
    style: { transform: CSS.Transform.toString(transform), transition },
  };
};
