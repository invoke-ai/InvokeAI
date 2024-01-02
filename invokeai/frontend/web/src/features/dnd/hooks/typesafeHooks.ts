import { useDraggable, useDroppable } from '@dnd-kit/core';
import type {
  UseDraggableTypesafeArguments,
  UseDraggableTypesafeReturnValue,
  UseDroppableTypesafeArguments,
  UseDroppableTypesafeReturnValue,
} from 'features/dnd/types';

export function useDroppableTypesafe(props: UseDroppableTypesafeArguments) {
  return useDroppable(props) as UseDroppableTypesafeReturnValue;
}

export function useDraggableTypesafe(props: UseDraggableTypesafeArguments) {
  return useDraggable(props) as UseDraggableTypesafeReturnValue;
}
