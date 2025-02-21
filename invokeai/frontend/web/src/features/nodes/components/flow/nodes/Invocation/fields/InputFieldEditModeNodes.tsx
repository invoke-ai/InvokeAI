import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Spacer } from '@invoke-ai/ui-library';
import { InputFieldAddToFormRoot } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldAddToFormRoot';
import { InputFieldDescriptionPopover } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldDescriptionPopover';
import { InputFieldHandle } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldHandle';
import { InputFieldResetToDefaultValueIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldResetToDefaultValueIconButton';
import { useNodeFieldDnd } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { useInputFieldIsConnected } from 'features/nodes/hooks/useInputFieldIsConnected';
import { useInputFieldIsInvalid } from 'features/nodes/hooks/useInputFieldIsInvalid';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import { NO_DRAG_CLASS } from 'features/nodes/types/constants';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useRef, useState } from 'react';

import { InputFieldRenderer } from './InputFieldRenderer';
import { InputFieldTitle } from './InputFieldTitle';
import { InputFieldWrapper } from './InputFieldWrapper';

interface Props {
  nodeId: string;
  fieldName: string;
}

export const InputFieldEditModeNodes = memo(({ nodeId, fieldName }: Props) => {
  const fieldTemplate = useInputFieldTemplate(nodeId, fieldName);
  const isInvalid = useInputFieldIsInvalid(nodeId, fieldName);
  const isConnected = useInputFieldIsConnected(nodeId, fieldName);

  if (fieldTemplate.input === 'connection' || isConnected) {
    return (
      <ConnectedOrConnectionField
        nodeId={nodeId}
        fieldName={fieldName}
        isInvalid={isInvalid}
        isConnected={isConnected}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  return (
    <DirectField
      nodeId={nodeId}
      fieldName={fieldName}
      isInvalid={isInvalid}
      isConnected={isConnected}
      fieldTemplate={fieldTemplate}
    />
  );
});

InputFieldEditModeNodes.displayName = 'InputFieldEditModeNodes';

type CommonProps = {
  nodeId: string;
  fieldName: string;
  isInvalid: boolean;
  isConnected: boolean;
  fieldTemplate: FieldInputTemplate;
};

const ConnectedOrConnectionField = memo(({ nodeId, fieldName, isInvalid }: CommonProps) => {
  return (
    <InputFieldWrapper>
      <Flex px={2}>
        <InputFieldTitle nodeId={nodeId} fieldName={fieldName} isInvalid={isInvalid} />
      </Flex>
      <InputFieldHandle nodeId={nodeId} fieldName={fieldName} />
    </InputFieldWrapper>
  );
});
ConnectedOrConnectionField.displayName = 'ConnectedOrConnectionField';

const directFieldSx: SystemStyleObject = {
  w: 'full',
  px: 2,
  flexDir: 'column',
  gap: 1,
  '&[data-is-dragging="true"]': {
    opacity: 0.3,
  },
  // Without pointerEvents prop, disabled inputs don't trigger reactflow events. For example, when making a
  // connection, the mouse up to end the connection won't fire, leaving the connection in-progress.
  pointerEvents: 'auto',
  '&[data-is-connected="true"]': {
    pointerEvents: 'none',
  },
};

const DirectField = memo(({ nodeId, fieldName, isInvalid, isConnected, fieldTemplate }: CommonProps) => {
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const [isHovered, setIsHovered] = useState(false);

  const onMouseEnter = useCallback(() => {
    setIsHovered(true);
  }, []);

  const onMouseLeave = useCallback(() => {
    setIsHovered(false);
  }, []);

  const isDragging = useNodeFieldDnd({ nodeId, fieldName }, fieldTemplate, draggableRef, dragHandleRef);

  return (
    <InputFieldWrapper>
      <Flex
        ref={draggableRef}
        sx={directFieldSx}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        data-is-connected={isConnected}
        data-is-dragging={isDragging}
      >
        <Flex gap={1}>
          <Flex className={NO_DRAG_CLASS} ref={dragHandleRef}>
            <InputFieldTitle nodeId={nodeId} fieldName={fieldName} isInvalid={isInvalid} isDragging={isDragging} />
          </Flex>
          <Spacer />
          {isHovered && (
            <>
              <InputFieldDescriptionPopover nodeId={nodeId} fieldName={fieldName} />
              <InputFieldResetToDefaultValueIconButton nodeId={nodeId} fieldName={fieldName} />
              <InputFieldAddToFormRoot nodeId={nodeId} fieldName={fieldName} />
            </>
          )}
        </Flex>
        <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
      </Flex>
      {fieldTemplate.input !== 'direct' && <InputFieldHandle nodeId={nodeId} fieldName={fieldName} />}
    </InputFieldWrapper>
  );
});
DirectField.displayName = 'DirectField';
