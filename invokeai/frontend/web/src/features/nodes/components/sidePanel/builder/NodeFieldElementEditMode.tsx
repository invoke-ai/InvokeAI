import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, FormControl } from '@invoke-ai/ui-library';
import { InputFieldRenderer } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { useFormElementDnd } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { FormElementEditModeContent } from 'features/nodes/components/sidePanel/builder/FormElementEditModeContent';
import { FormElementEditModeHeader } from 'features/nodes/components/sidePanel/builder/FormElementEditModeHeader';
import { NodeFieldElementDescriptionEditable } from 'features/nodes/components/sidePanel/builder/NodeFieldElementDescriptionEditable';
import { NodeFieldElementLabelEditable } from 'features/nodes/components/sidePanel/builder/NodeFieldElementLabelEditable';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { NODE_FIELD_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo, useRef } from 'react';

const sx: SystemStyleObject = {
  position: 'relative',
  borderRadius: 'base',
  '&[data-parent-layout="column"]': {
    w: 'full',
    h: 'min-content',
  },
  '&[data-parent-layout="row"]': {
    flex: '1 1 0',
  },
  flexDir: 'column',
};

export const NodeFieldElementEditMode = memo(({ el }: { el: NodeFieldElement }) => {
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const [activeDropRegion, isDragging] = useFormElementDnd(el.id, draggableRef, dragHandleRef);
  const containerCtx = useContainerContext();
  const { id, data } = el;
  const { fieldIdentifier, showDescription } = data;

  return (
    <Flex ref={draggableRef} id={id} className={NODE_FIELD_CLASS_NAME} sx={sx} data-parent-layout={containerCtx.layout}>
      <FormElementEditModeHeader dragHandleRef={dragHandleRef} element={el} data-is-dragging={isDragging} />
      <FormElementEditModeContent data-is-dragging={isDragging} p={4}>
        <FormControl flex="1 1 0" orientation="vertical">
          <NodeFieldElementLabelEditable el={el} />
          <Flex w="full" gap={4}>
            <InputFieldRenderer
              nodeId={fieldIdentifier.nodeId}
              fieldName={fieldIdentifier.fieldName}
              settings={data.settings}
            />
          </Flex>
          {showDescription && <NodeFieldElementDescriptionEditable el={el} />}
        </FormControl>
      </FormElementEditModeContent>
      <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
    </Flex>
  );
});

NodeFieldElementEditMode.displayName = 'NodeFieldElementEditMode';
