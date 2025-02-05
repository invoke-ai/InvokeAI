import { Flex, FormControl, FormHelperText, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldRenderer } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { useInputFieldDescription } from 'features/nodes/hooks/useInputFieldDescription';
import { useInputFieldLabel } from 'features/nodes/hooks/useInputFieldLabel';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import { selectWorkflowFormMode, useElement } from 'features/nodes/store/workflowSlice';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { isNodeFieldElement, NODE_FIELD_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo, useMemo } from 'react';

export const NodeFieldElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowFormMode);

  if (!el || !isNodeFieldElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return (
      <InputFieldGate nodeId={el.data.fieldIdentifier.nodeId} fieldName={el.data.fieldIdentifier.fieldName}>
        <NodeFieldElementComponentViewMode el={el} />
      </InputFieldGate>
    );
  }

  // mode === 'edit'
  return (
    <InputFieldGate nodeId={el.data.fieldIdentifier.nodeId} fieldName={el.data.fieldIdentifier.fieldName}>
      <NodeFieldElementComponentEditMode el={el} />{' '}
    </InputFieldGate>
  );
});

NodeFieldElementComponent.displayName = 'NodeFieldElementComponent';

export const NodeFieldElementInputComponent = memo(({ el }: { el: NodeFieldElement }) => {
  const { data } = el;
  const { fieldIdentifier, showLabel, showDescription } = data;
  const label = useInputFieldLabel(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const description = useInputFieldDescription(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const fieldTemplate = useInputFieldTemplate(fieldIdentifier.nodeId, fieldIdentifier.fieldName);

  const _label = useMemo(() => label || fieldTemplate.title, [label, fieldTemplate.title]);
  const _description = useMemo(
    () => description || fieldTemplate.description,
    [description, fieldTemplate.description]
  );

  return (
    <FormControl flex="1 1 0" orientation="vertical">
      {showLabel && _label && <FormLabel>{_label}</FormLabel>}
      <Flex w="full" gap={4}>
        <InputFieldRenderer
          nodeId={fieldIdentifier.nodeId}
          fieldName={fieldIdentifier.fieldName}
          config={data.config}
        />
      </Flex>
      {showDescription && _description && <FormHelperText>{_description}</FormHelperText>}
    </FormControl>
  );
});
NodeFieldElementInputComponent.displayName = 'NodeFieldElementInputComponent';

export const NodeFieldElementComponentViewMode = memo(({ el }: { el: NodeFieldElement }) => {
  const { id } = el;

  return (
    <Flex id={id} className={NODE_FIELD_CLASS_NAME} flex="1 1 0">
      <NodeFieldElementInputComponent el={el} />
    </Flex>
  );
});

NodeFieldElementComponentViewMode.displayName = 'NodeFieldElementComponentViewMode';

export const NodeFieldElementComponentEditMode = memo(({ el }: { el: NodeFieldElement }) => {
  const { id } = el;

  return (
    <FormElementEditModeWrapper element={el}>
      <Flex id={id} className={NODE_FIELD_CLASS_NAME} flex="1 1 0">
        <NodeFieldElementInputComponent el={el} />
      </Flex>
    </FormElementEditModeWrapper>
  );
});

NodeFieldElementComponentEditMode.displayName = 'NodeFieldElementComponentEditMode';
