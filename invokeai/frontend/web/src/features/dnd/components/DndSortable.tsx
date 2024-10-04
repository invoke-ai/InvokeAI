import type { DragEndEvent } from '@dnd-kit/core';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

import { DndContextTypesafe } from './DndContextTypesafe';

type Props = PropsWithChildren & {
  items: string[];
  onDragEnd(event: DragEndEvent): void;
};

const DndSortable = (props: Props) => {
  return (
    <DndContextTypesafe onDragEnd={props.onDragEnd}>
      <SortableContext items={props.items} strategy={verticalListSortingStrategy}>
        {props.children}
      </SortableContext>
    </DndContextTypesafe>
  );
};

export default memo(DndSortable);
