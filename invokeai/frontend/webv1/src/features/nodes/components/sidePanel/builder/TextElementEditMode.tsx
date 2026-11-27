import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { useFormElementDnd } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { FormElementEditModeContent } from 'features/nodes/components/sidePanel/builder/FormElementEditModeContent';
import { FormElementEditModeHeader } from 'features/nodes/components/sidePanel/builder/FormElementEditModeHeader';
import { TEXT_CLASS_NAME, type TextElement } from 'features/nodes/types/workflow';
import { memo, useRef } from 'react';

import { TextElementContentEditable } from './TextElementContentEditable';

const sx: SystemStyleObject = {
  position: 'relative',
  borderRadius: 'base',
  minW: 32,
  '&[data-parent-layout="column"]': {
    w: 'full',
    h: 'min-content',
  },
  '&[data-parent-layout="row"]': {
    flex: '1 0 0',
  },
  flexDir: 'column',
};

export const TextElementEditMode = memo(({ el }: { el: TextElement }) => {
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const [activeDropRegion, isDragging] = useFormElementDnd(el.id, draggableRef, dragHandleRef);
  const containerCtx = useContainerContext();
  const { id } = el;

  return (
    <Flex ref={draggableRef} id={id} className={TEXT_CLASS_NAME} sx={sx} data-parent-layout={containerCtx.layout}>
      <FormElementEditModeHeader dragHandleRef={dragHandleRef} element={el} data-is-dragging={isDragging} />
      <FormElementEditModeContent data-is-dragging={isDragging} p={4}>
        <TextElementContentEditable el={el} />
      </FormElementEditModeContent>
      <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
    </Flex>
  );
});

TextElementEditMode.displayName = 'TextElementEditMode';
