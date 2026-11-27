import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { DividerElementComponent } from 'features/nodes/components/sidePanel/builder/DividerElementComponent';
import { useFormElementDnd } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { FormElementEditModeContent } from 'features/nodes/components/sidePanel/builder/FormElementEditModeContent';
import { FormElementEditModeHeader } from 'features/nodes/components/sidePanel/builder/FormElementEditModeHeader';
import type { DividerElement } from 'features/nodes/types/workflow';
import { DIVIDER_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo, useRef } from 'react';

const sx: SystemStyleObject = {
  position: 'relative',
  borderRadius: 'base',
  '&[data-parent-layout="column"]': {
    w: 'full',
    h: 'min-content',
  },
  '&[data-parent-layout="row"]': {
    w: 'min-content',
    h: 'full',
  },
  flexDir: 'column',
};

export const DividerElementEditMode = memo(({ el }: { el: DividerElement }) => {
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const [activeDropRegion, isDragging] = useFormElementDnd(el.id, draggableRef, dragHandleRef);
  const containerCtx = useContainerContext();
  const { id } = el;

  return (
    <Flex ref={draggableRef} id={id} className={DIVIDER_CLASS_NAME} sx={sx} data-parent-layout={containerCtx.layout}>
      <FormElementEditModeHeader dragHandleRef={dragHandleRef} element={el} data-is-dragging={isDragging} />
      <FormElementEditModeContent data-is-dragging={isDragging} p={4}>
        <DividerElementComponent />
      </FormElementEditModeContent>
      <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
    </Flex>
  );
});

DividerElementEditMode.displayName = 'DividerElementEditMode';
