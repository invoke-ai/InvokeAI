import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { useFormElementDnd } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { FormElementEditModeHeader } from 'features/nodes/components/sidePanel/builder/FormElementEditModeHeader';
import { getEditModeWrapperId } from 'features/nodes/components/sidePanel/builder/shared';
import type { FormElement } from 'features/nodes/types/workflow';
import type { PropsWithChildren } from 'react';
import { memo, useRef } from 'react';

import { useIsRootElement } from './dnd-hooks';

const EDIT_MODE_WRAPPER_CLASS_NAME = getPrefixedId('edit-mode-wrapper', '-');

const wrapperSx: SystemStyleObject = {
  position: 'relative',
  flex: '1 1 0',
  '&[data-element-type="divider"]&[data-layout="row"]': {
    flex: '0 1 0',
  },
  borderRadius: 'base',
};

const innerSx: SystemStyleObject = {
  position: 'relative',
  flexDir: 'column',
  alignItems: 'center',
  justifyContent: 'flex-start',
  borderRadius: 'base',
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
  const [activeDropRegion, isDragging] = useFormElementDnd(element.id, draggableRef, dragHandleRef);
  const containerCtx = useContainerContext();
  const isRootElement = useIsRootElement(element.id);

  return (
    <Flex
      id={getEditModeWrapperId(element.id)}
      ref={draggableRef}
      className={EDIT_MODE_WRAPPER_CLASS_NAME}
      sx={wrapperSx}
      data-is-root={isRootElement}
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
        <Flex
          w="full"
          h="full"
          p={4}
          gap={4}
          borderWidth={1}
          borderColor="base.800"
          borderRadius="base"
          borderTopRadius="unset"
          borderTop="unset"
        >
          {children}
        </Flex>
      </Flex>
      <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
    </Flex>
  );
});

FormElementEditModeWrapper.displayName = 'FormElementEditModeWrapper';
