import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { FormElementEditModeHeader } from 'features/nodes/components/sidePanel/builder/FormElementEditModeHeader';
import { EDIT_MODE_WRAPPER_CLASS_NAME, getEditModeWrapperId } from 'features/nodes/components/sidePanel/builder/shared';
import { useDraggableFormElement } from 'features/nodes/components/sidePanel/builder/use-builder-dnd';
import type { FormElement } from 'features/nodes/types/workflow';
import type { PropsWithChildren } from 'react';
import { memo, useRef } from 'react';

const sx: SystemStyleObject = {
  position: 'relative',
  flexDir: 'column',
  boxShadow: '0 0 0 1px var(--invoke-colors-base-750)',
  borderRadius: 'base',
  alignItems: 'center',
  justifyContent: 'flex-start',
  w: 'full',
  h: 'full',
  '&[data-is-dragging="true"]': {
    opacity: 0.3,
  },
  '&[data-active-drop-region="center"]': {
    opacity: 1,
    bg: 'base.700',
  },
};

export const FormElementEditModeWrapper = memo(({ element, children }: PropsWithChildren<{ element: FormElement }>) => {
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const [activeDropRegion, isDragging] = useDraggableFormElement(element.id, draggableRef, dragHandleRef);

  return (
    <Flex
      id={getEditModeWrapperId(element.id)}
      ref={draggableRef}
      className={EDIT_MODE_WRAPPER_CLASS_NAME}
      position="relative"
      flex={1}
    >
      <Flex sx={sx} data-is-dragging={isDragging} data-active-drop-region={activeDropRegion}>
        <FormElementEditModeHeader ref={dragHandleRef} element={element} />
        <Flex w="full" p={4} alignItems="center" gap={4}>
          {children}
        </Flex>
      </Flex>
      <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
    </Flex>
  );
});

FormElementEditModeWrapper.displayName = 'FormElementEditModeWrapper';
