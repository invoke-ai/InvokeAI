import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Divider, Flex, FormControl } from '@invoke-ai/ui-library';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldRenderer } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { useFormElementDnd } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { FormElementEditModeContent } from 'features/nodes/components/sidePanel/builder/FormElementEditModeContent';
import { FormElementEditModeHeader } from 'features/nodes/components/sidePanel/builder/FormElementEditModeHeader';
import { NodeFieldElementDescriptionEditable } from 'features/nodes/components/sidePanel/builder/NodeFieldElementDescriptionEditable';
import { NodeFieldElementLabelEditable } from 'features/nodes/components/sidePanel/builder/NodeFieldElementLabelEditable';
import { NodeFieldElementStringDropdownSettings } from 'features/nodes/components/sidePanel/builder/NodeFieldElementStringDropdownSettings';
import { useMouseOverFormField, useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { NODE_FIELD_CLASS_NAME } from 'features/nodes/types/workflow';
import type { RefObject } from 'react';
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
  const { id } = el;

  return (
    <Flex ref={draggableRef} id={id} className={NODE_FIELD_CLASS_NAME} sx={sx} data-parent-layout={containerCtx.layout}>
      <NodeFieldElementEditModeContent dragHandleRef={dragHandleRef} el={el} isDragging={isDragging} />
      <NodeFieldElementOverlay nodeId={el.data.fieldIdentifier.nodeId} />
      <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
    </Flex>
  );
});

NodeFieldElementEditMode.displayName = 'NodeFieldElementEditMode';

const NodeFieldElementEditModeContent = memo(
  ({
    el,
    dragHandleRef,
    isDragging,
  }: {
    el: NodeFieldElement;
    dragHandleRef: RefObject<HTMLDivElement>;
    isDragging: boolean;
  }) => {
    const { id, data } = el;
    const { fieldIdentifier, showDescription } = data;
    return (
      <>
        <FormElementEditModeHeader dragHandleRef={dragHandleRef} element={el} data-is-dragging={isDragging} />
        <FormElementEditModeContent data-is-dragging={isDragging} p={4}>
          <InputFieldGate nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName}>
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
              {data.settings?.type === 'string-field-config' && data.settings.component === 'dropdown' && (
                <>
                  <Divider />
                  <NodeFieldElementStringDropdownSettings id={id} settings={data.settings} />
                </>
              )}
            </FormControl>
          </InputFieldGate>
        </FormElementEditModeContent>
      </>
    );
  }
);
NodeFieldElementEditModeContent.displayName = 'NodeFieldElementEditModeContent';

const nodeFieldOverlaySx: SystemStyleObject = {
  position: 'absolute',
  top: 0,
  insetInlineEnd: 0,
  bottom: 0,
  insetInlineStart: 0,
  borderRadius: 'base',
  transitionProperty: 'none',
  pointerEvents: 'none',
  display: 'none',
  '&[data-is-mouse-over-node-or-form-field="true"]': {
    display: 'block',
    bg: 'invokeBlueAlpha.100',
  },
};

export const NodeFieldElementOverlay = memo(({ nodeId }: { nodeId: string }) => {
  const mouseOverNode = useMouseOverNode(nodeId);
  const mouseOverFormField = useMouseOverFormField(nodeId);

  return (
    <Box
      sx={nodeFieldOverlaySx}
      data-is-mouse-over-node-or-form-field={mouseOverNode.isMouseOverNode || mouseOverFormField.isMouseOverFormField}
    />
  );
});
NodeFieldElementOverlay.displayName = 'NodeFieldElementOverlay';
