import { Flex, FormControl, FormHelperText, FormLabel, Input, Spacer, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldRenderer } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { NodeFieldElementResetToInitialValueIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/NodeFieldElementResetToInitialValueIconButton';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { useInputFieldDescription } from 'features/nodes/hooks/useInputFieldDescription';
import { useInputFieldLabel } from 'features/nodes/hooks/useInputFieldLabel';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import { fieldDescriptionChanged, fieldLabelChanged } from 'features/nodes/store/nodesSlice';
import { selectWorkflowMode, useElement } from 'features/nodes/store/workflowSlice';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { isNodeFieldElement, NODE_FIELD_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo, useCallback, useMemo, useRef } from 'react';

export const NodeFieldElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);

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

const NodeFieldElementComponentViewMode = memo(({ el }: { el: NodeFieldElement }) => {
  const { id, data } = el;
  const { fieldIdentifier, showDescription } = data;
  const label = useInputFieldLabel(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const description = useInputFieldDescription(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const fieldTemplate = useInputFieldTemplate(fieldIdentifier.nodeId, fieldIdentifier.fieldName);

  const _label = useMemo(() => label || fieldTemplate.title, [label, fieldTemplate.title]);
  const _description = useMemo(
    () => description || fieldTemplate.description,
    [description, fieldTemplate.description]
  );

  return (
    <Flex id={id} className={NODE_FIELD_CLASS_NAME} flex={1}>
      <FormControl flex="1 1 0" orientation="vertical">
        <Flex w="full" gap={4}>
          <FormLabel>{_label}</FormLabel>
          <Spacer />
          <NodeFieldElementResetToInitialValueIconButton element={el} />
        </Flex>
        <Flex w="full" gap={4}>
          <InputFieldRenderer
            nodeId={fieldIdentifier.nodeId}
            fieldName={fieldIdentifier.fieldName}
            settings={data.settings}
          />
        </Flex>
        {showDescription && _description && <FormHelperText>{_description}</FormHelperText>}
      </FormControl>
    </Flex>
  );
});

NodeFieldElementComponentViewMode.displayName = 'NodeFieldElementComponentViewMode';

const NodeFieldElementComponentEditMode = memo(({ el }: { el: NodeFieldElement }) => {
  const { id, data } = el;
  const { fieldIdentifier, showDescription } = data;

  return (
    <FormElementEditModeWrapper element={el}>
      <Flex id={id} className={NODE_FIELD_CLASS_NAME} flex="1 1 0">
        <FormControl flex="1 1 0" orientation="vertical">
          <NodeFieldEditableLabel el={el} />
          <Flex w="full" gap={4}>
            <InputFieldRenderer
              nodeId={fieldIdentifier.nodeId}
              fieldName={fieldIdentifier.fieldName}
              settings={data.settings}
            />
          </Flex>
          {showDescription && <NodeFieldEditableDescription el={el} />}
        </FormControl>
      </Flex>
    </FormElementEditModeWrapper>
  );
});

NodeFieldElementComponentEditMode.displayName = 'NodeFieldElementComponentEditMode';

const NodeFieldEditableLabel = memo(({ el }: { el: NodeFieldElement }) => {
  const { data } = el;
  const { fieldIdentifier } = data;
  const dispatch = useAppDispatch();
  const label = useInputFieldLabel(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const fieldTemplate = useInputFieldTemplate(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const inputRef = useRef<HTMLInputElement>(null);

  const onChange = useCallback(
    (label: string) => {
      dispatch(fieldLabelChanged({ nodeId: fieldIdentifier.nodeId, fieldName: fieldIdentifier.fieldName, label }));
    },
    [dispatch, fieldIdentifier.fieldName, fieldIdentifier.nodeId]
  );

  const editable = useEditable({
    value: label || fieldTemplate.title,
    defaultValue: fieldTemplate.title,
    inputRef,
    onChange,
  });

  if (!editable.isEditing) {
    return (
      <Flex w="full" gap={4}>
        <FormLabel onDoubleClick={editable.startEditing} cursor="text">
          {editable.value}
        </FormLabel>
        <Spacer />
        <NodeFieldElementResetToInitialValueIconButton element={el} />
      </Flex>
    );
  }

  return (
    <Input
      ref={inputRef}
      variant="outline"
      p={1}
      px={2}
      _focusVisible={{ borderRadius: 'base', h: 'unset' }}
      {...editable.inputProps}
    />
  );
});
NodeFieldEditableLabel.displayName = 'NodeFieldEditableLabel';

const NodeFieldEditableDescription = memo(({ el }: { el: NodeFieldElement }) => {
  const { data } = el;
  const { fieldIdentifier } = data;
  const dispatch = useAppDispatch();
  const description = useInputFieldDescription(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const fieldTemplate = useInputFieldTemplate(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const onChange = useCallback(
    (description: string) => {
      dispatch(
        fieldDescriptionChanged({
          nodeId: fieldIdentifier.nodeId,
          fieldName: fieldIdentifier.fieldName,
          val: description,
        })
      );
    },
    [dispatch, fieldIdentifier.fieldName, fieldIdentifier.nodeId]
  );

  const editable = useEditable({
    value: description || fieldTemplate.description,
    defaultValue: fieldTemplate.description,
    inputRef,
    onChange,
  });

  if (!editable.isEditing) {
    return <FormHelperText onDoubleClick={editable.startEditing}>{editable.value}</FormHelperText>;
  }

  return (
    <Textarea
      ref={inputRef}
      variant="outline"
      fontSize="sm"
      p={1}
      px={2}
      _focusVisible={{ borderRadius: 'base', h: 'unset' }}
      {...editable.inputProps}
    />
  );
});
NodeFieldEditableDescription.displayName = 'NodeFieldEditableDescription';
