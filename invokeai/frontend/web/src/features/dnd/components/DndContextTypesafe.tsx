import { DndContext } from '@dnd-kit/core';
import type { DndContextTypesafeProps } from 'features/dnd/types';

export function DndContextTypesafe(props: DndContextTypesafeProps) {
  return <DndContext {...props} />;
}
