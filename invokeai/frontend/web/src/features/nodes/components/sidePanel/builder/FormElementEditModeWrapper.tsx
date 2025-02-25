import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useContainerContext, useDepthContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { useFormElementDnd } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { FormElementEditModeHeader } from 'features/nodes/components/sidePanel/builder/FormElementEditModeHeader';
import { getEditModeWrapperId } from 'features/nodes/components/sidePanel/builder/shared';
import type { FormElement } from 'features/nodes/types/workflow';
import type { PropsWithChildren } from 'react';
import { memo, useRef } from 'react';

export const EDIT_MODE_WRAPPER_CLASS_NAME = 'edit-mode-wrapper';

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
    bg: 'base.850',
  },
  '&[data-element-type="divider"]&[data-layout="row"]': {
    w: 'min-content',
  },
  '&[data-element-type="divider"]&[data-layout="column"]': {
    h: 'min-content',
  },
};

const contentWrapperSx: SystemStyleObject = {
  w: 'full',
  h: 'full',
  p: 4,
  gap: 4,
  borderWidth: 1,
  borderRadius: 'base',
  borderTopRadius: 'unset',
  borderTop: 'unset',
  borderColor: 'baseAlpha.250',
  '&[data-depth="0"]': { borderColor: 'baseAlpha.100' },
  '&[data-depth="1"]': { borderColor: 'baseAlpha.150' },
  '&[data-depth="2"]': { borderColor: 'baseAlpha.200' },
};

export const FormElementEditModeWrapper = memo(({ element, children }: PropsWithChildren<{ element: FormElement }>) => {
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const [activeDropRegion, isDragging] = useFormElementDnd(element.id, draggableRef, dragHandleRef);
  const containerCtx = useContainerContext();
  const depth = useDepthContext();

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
        <Flex sx={contentWrapperSx} data-depth={depth}>
          {children}
        </Flex>
      </Flex>
      <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
    </Flex>
  );
});

FormElementEditModeWrapper.displayName = 'FormElementEditModeWrapper';
