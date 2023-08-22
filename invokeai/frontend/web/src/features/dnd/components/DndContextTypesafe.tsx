import { DndContext } from '@dnd-kit/core';
import { DndContextTypesafeProps } from '../types';

export function DndContextTypesafe(props: DndContextTypesafeProps) {
  return <DndContext {...props} />;
}
