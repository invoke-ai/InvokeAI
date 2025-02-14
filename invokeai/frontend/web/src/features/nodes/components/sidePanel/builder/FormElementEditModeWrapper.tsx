import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { FormElementEditModeHeader } from 'features/nodes/components/sidePanel/builder/FormElementEditModeHeader';
import { EDIT_MODE_WRAPPER_CLASS_NAME, getEditModeWrapperId } from 'features/nodes/components/sidePanel/builder/shared';
import { useDraggableFormElement } from 'features/nodes/components/sidePanel/builder/use-builder-dnd';
import type { FormElement } from 'features/nodes/types/workflow';
import type { PropsWithChildren } from 'react';
import { memo, useRef } from 'react';

const wrapperSx: SystemStyleObject = {
  position: 'relative',
  flex: '1 1 0',
  '&[data-element-type="divider"]&[data-layout="row"]': {
    flex: '0 1 0',
  },
};

const innerSx: SystemStyleObject = {
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
  '&[data-element-type="divider"]&[data-layout="row"]': {
    w: 'min-content',
  },
  '&[data-element-type="divider"]&[data-layout="column"]': {
    h: 'min-content',
  },
};

export const FormElementEditModeWrapper = memo(({ element, children }: PropsWithChildren<{ element: FormElement }>) => {
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const [activeDropRegion, isDragging] = useDraggableFormElement(element.id, draggableRef, dragHandleRef);
  const containerCtx = useContainerContext();

  return (
    <Flex
      id={getEditModeWrapperId(element.id)}
      ref={draggableRef}
      className={EDIT_MODE_WRAPPER_CLASS_NAME}
      sx={wrapperSx}
      data-element-type={element.type}
      data-layout={containerCtx?.layout}
    >
      <Flex
        sx={innerSx}
        data-is-dragging={isDragging}
        data-active-drop-region={activeDropRegion}
        data-element-type={element.type}
        data-layout={containerCtx?.layout}
      >
        <FormElementEditModeHeader ref={dragHandleRef} element={element} />
        <Flex w="full" h="full" p={4} gap={4}>
          {children}
        </Flex>
      </Flex>
      <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
    </Flex>
  );
});

FormElementEditModeWrapper.displayName = 'FormElementEditModeWrapper';
