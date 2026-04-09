import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Box, Flex, Grid, GridItem, Text, Tooltip } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { Handle, Position } from '@xyflow/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { getFieldColor } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { FloatFieldInput } from 'features/nodes/components/flow/nodes/Invocation/fields/FloatField/FloatFieldInput';
import { FloatFieldInputAndSlider } from 'features/nodes/components/flow/nodes/Invocation/fields/FloatField/FloatFieldInputAndSlider';
import { FloatFieldSlider } from 'features/nodes/components/flow/nodes/Invocation/fields/FloatField/FloatFieldSlider';
import { InputFieldEditModeNodes } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldEditModeNodes';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldWrapper } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldWrapper';
import BoardFieldInputComponent from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/BoardFieldInputComponent';
import BooleanFieldInputComponent from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/BooleanFieldInputComponent';
import ColorFieldInputComponent from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ColorFieldInputComponent';
import EnumFieldInputComponent from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/EnumFieldInputComponent';
import ModelIdentifierFieldInputComponent from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelIdentifierFieldInputComponent';
import SchedulerFieldInputComponent from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/SchedulerFieldInputComponent';
import StylePresetFieldInputComponent from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/StylePresetFieldInputComponent';
import { IntegerFieldInput } from 'features/nodes/components/flow/nodes/Invocation/fields/IntegerField/IntegerFieldInput';
import { IntegerFieldInputAndSlider } from 'features/nodes/components/flow/nodes/Invocation/fields/IntegerField/IntegerFieldInputAndSlider';
import { IntegerFieldSlider } from 'features/nodes/components/flow/nodes/Invocation/fields/IntegerField/IntegerFieldSlider';
import { OutputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldGate';
import { OutputFieldNodesEditorView } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldNodesEditorView';
import { StringFieldDropdown } from 'features/nodes/components/flow/nodes/Invocation/fields/StringField/StringFieldDropdown';
import { StringFieldInput } from 'features/nodes/components/flow/nodes/Invocation/fields/StringField/StringFieldInput';
import { StringFieldTextarea } from 'features/nodes/components/flow/nodes/Invocation/fields/StringField/StringFieldTextarea';
import InvocationNodeFooter from 'features/nodes/components/flow/nodes/Invocation/InvocationNodeFooter';
import InvocationNodeHeader from 'features/nodes/components/flow/nodes/Invocation/InvocationNodeHeader';
import { useInputFieldInstance } from 'features/nodes/hooks/useInputFieldInstance';
import { useOutputFieldNames } from 'features/nodes/hooks/useOutputFieldNames';
import { useFieldTypeName } from 'features/nodes/hooks/usePrettyFieldType';
import { useWithFooter } from 'features/nodes/hooks/useWithFooter';
import { $templates, callSavedWorkflowDynamicFieldsChanged } from 'features/nodes/store/nodesSlice';
import type {
  BoardFieldInputInstance,
  BoardFieldInputTemplate,
  BooleanFieldInputInstance,
  BooleanFieldInputTemplate,
  ColorFieldInputInstance,
  ColorFieldInputTemplate,
  EnumFieldInputInstance,
  EnumFieldInputTemplate,
  FieldInputInstance,
  FieldInputTemplate,
  FloatFieldInputInstance,
  FloatFieldInputTemplate,
  IntegerFieldInputInstance,
  IntegerFieldInputTemplate,
  ModelIdentifierFieldInputInstance,
  ModelIdentifierFieldInputTemplate,
  SavedWorkflowFieldInputInstance,
  SchedulerFieldInputInstance,
  SchedulerFieldInputTemplate,
  StringFieldInputInstance,
  StringFieldInputTemplate,
  StylePresetFieldInputInstance,
  StylePresetFieldInputTemplate,
} from 'features/nodes/types/field';
import {
  isBoardFieldInputInstance,
  isBooleanFieldInputInstance,
  isColorFieldInputInstance,
  isEnumFieldInputInstance,
  isFloatFieldInputInstance,
  isIntegerFieldInputInstance,
  isModelFieldType,
  isModelIdentifierFieldInputInstance,
  isSchedulerFieldInputInstance,
  isStringFieldInputInstance,
  isStylePresetFieldInputInstance,
} from 'features/nodes/types/field';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import type { CSSProperties } from 'react';
import { memo, useEffect, useMemo } from 'react';
import { useGetWorkflowQuery } from 'services/api/endpoints/workflows';

import { getSavedWorkflowDynamicFields } from './callSavedWorkflowFormUtils';

const bodySx: SystemStyleObject = {
  flexDirection: 'column',
  w: 'full',
  h: 'full',
  py: 2,
  gap: 1,
  borderBottomRadius: 'base',
  '&[data-with-footer="true"]': {
    borderBottomRadius: 0,
  },
  '&[data-with-footer="false"]': {
    pb: 4,
  },
};

const dynamicFieldSx: SystemStyleObject = {
  px: 2,
  flexDir: 'column',
  w: 'full',
};

const fieldBodySx: SystemStyleObject = {
  px: 2,
  py: 1,
  gap: 1,
  flexDir: 'column',
  w: 'full',
  pointerEvents: 'auto',
};

const handleSx: SystemStyleObject = {
  position: 'relative',
  width: 'full',
  height: 'full',
  borderStyle: 'solid',
  borderWidth: 4,
  pointerEvents: 'none',
  '&[data-cardinality="SINGLE"]': {
    borderWidth: 0,
  },
  borderRadius: '100%',
  '&[data-is-model-field="true"], &[data-is-batch-field="true"]': {
    borderRadius: 4,
  },
  '&[data-is-batch-field="true"]': {
    transform: 'rotate(45deg)',
  },
};

const handleStyles = {
  position: 'absolute',
  width: '1rem',
  height: '1rem',
  zIndex: 1,
  background: 'none',
  border: 'none',
  insetInlineStart: '-0.5rem',
  top: '50%',
  transform: 'translateY(-50%)',
} satisfies CSSProperties;

type Props = {
  nodeId: string;
  isOpen: boolean;
};

const CallSavedWorkflowNode = ({ nodeId, isOpen }: Props) => {
  const withFooter = useWithFooter();
  const workflowIdField = useInputFieldInstance<SavedWorkflowFieldInputInstance>('workflow_id');
  const templates = useStore($templates);
  const dispatch = useAppDispatch();

  const { data: workflow } = useGetWorkflowQuery(workflowIdField.value, {
    skip: !workflowIdField.value,
  });

  const dynamicFields = useMemo(() => getSavedWorkflowDynamicFields(workflow, templates), [templates, workflow]);

  useEffect(() => {
    dispatch(callSavedWorkflowDynamicFieldsChanged({ nodeId, fields: dynamicFields }));
  }, [dispatch, dynamicFields, nodeId]);

  return (
    <>
      <InvocationNodeHeader nodeId={nodeId} isOpen={isOpen} />
      {isOpen && (
        <>
          <Flex layerStyle="nodeBody" sx={bodySx} data-with-footer={withFooter}>
            <Flex flexDir="column" px={2} w="full" h="full">
              <Grid gridTemplateColumns="1fr auto" gridAutoRows="1fr">
                <GridItem gridColumnStart={1} gridRowStart={1}>
                  <InputFieldGate nodeId={nodeId} fieldName="workflow_id">
                    <InputFieldEditModeNodes nodeId={nodeId} fieldName="workflow_id" />
                  </InputFieldGate>
                </GridItem>
                <OutputFields nodeId={nodeId} />
              </Grid>
              <DynamicFieldsSection nodeId={nodeId} fields={dynamicFields} />
            </Flex>
          </Flex>
          {withFooter && <InvocationNodeFooter nodeId={nodeId} />}
        </>
      )}
    </>
  );
};

export default memo(CallSavedWorkflowNode);

const DynamicFieldsSection = memo(
  ({ nodeId, fields }: { nodeId: string; fields: ReturnType<typeof getSavedWorkflowDynamicFields> }) => {
    if (fields.length === 0) {
      return (
        <Badge variant="subtle" alignSelf="flex-start">
          Select a workflow with exposed form fields
        </Badge>
      );
    }

    return (
      <>
        {fields.map((field) => (
          <DynamicFieldRow key={field.fieldName} nodeId={nodeId} field={field} />
        ))}
      </>
    );
  }
);
DynamicFieldsSection.displayName = 'DynamicFieldsSection';

const DynamicFieldRow = memo(
  ({ nodeId, field }: { nodeId: string; field: ReturnType<typeof getSavedWorkflowDynamicFields>[number] }) => {
    const ctx = useInvocationNodeContext();
    const selector = useMemo(() => ctx.buildSelectInputFieldSafe(field.fieldName), [ctx, field.fieldName]);
    const instance = useAppSelector(selector);

    if (!instance) {
      return null;
    }

    return (
      <InputFieldWrapper>
        <Flex sx={dynamicFieldSx}>
          <Flex sx={fieldBodySx}>
            <Text fontSize="sm" fontWeight="semibold" color="base.300" noOfLines={1}>
              {instance.label || field.fieldTemplate.title}
            </Text>
            <DynamicFieldInputRenderer
              nodeId={nodeId}
              instance={instance}
              template={field.fieldTemplate}
              settings={field.settings}
            />
          </Flex>
        </Flex>
        <DynamicInputFieldHandle fieldName={field.fieldName} template={field.fieldTemplate} />
      </InputFieldWrapper>
    );
  }
);
DynamicFieldRow.displayName = 'DynamicFieldRow';

const DynamicInputFieldHandle = memo(({ fieldName, template }: { fieldName: string; template: FieldInputTemplate }) => {
  const fieldTypeName = useFieldTypeName(template.type);
  const fieldColor = useMemo(() => getFieldColor(template.type), [template.type]);
  const isModelField = useMemo(() => isModelFieldType(template.type), [template.type]);

  return (
    <Tooltip label={fieldTypeName} placement="start">
      <Handle type="target" id={fieldName} position={Position.Left} style={handleStyles}>
        <Box
          sx={handleSx}
          data-cardinality={template.type.cardinality}
          data-is-batch-field={template.type.batch}
          data-is-model-field={isModelField}
          backgroundColor={template.type.cardinality === 'SINGLE' ? fieldColor : 'base.900'}
          borderColor={fieldColor}
        />
      </Handle>
    </Tooltip>
  );
});
DynamicInputFieldHandle.displayName = 'DynamicInputFieldHandle';

const DynamicFieldInputRenderer = memo(
  ({
    nodeId,
    instance,
    template,
    settings,
  }: {
    nodeId: string;
    instance: FieldInputInstance;
    template: FieldInputTemplate;
    settings: NodeFieldElement['data']['settings'];
  }) => {
    if (template.type.name === 'StringField' && isStringFieldInputInstance(instance)) {
      if (settings?.type === 'string-field-config' && settings.component === 'textarea') {
        return (
          <StringFieldTextarea
            nodeId={nodeId}
            field={instance as StringFieldInputInstance}
            fieldTemplate={template as StringFieldInputTemplate}
          />
        );
      }
      if (settings?.type === 'string-field-config' && settings.component === 'dropdown') {
        return (
          <StringFieldDropdown
            nodeId={nodeId}
            field={instance as StringFieldInputInstance}
            fieldTemplate={template as StringFieldInputTemplate}
            settings={settings}
          />
        );
      }
      return (
        <StringFieldInput
          nodeId={nodeId}
          field={instance as StringFieldInputInstance}
          fieldTemplate={template as StringFieldInputTemplate}
        />
      );
    }

    if (template.type.name === 'IntegerField' && isIntegerFieldInputInstance(instance)) {
      if (settings?.type === 'integer-field-config' && settings.component === 'slider') {
        return (
          <IntegerFieldSlider
            nodeId={nodeId}
            field={instance as IntegerFieldInputInstance}
            fieldTemplate={template as IntegerFieldInputTemplate}
            settings={settings}
          />
        );
      }
      if (settings?.type === 'integer-field-config' && settings.component === 'number-input-and-slider') {
        return (
          <IntegerFieldInputAndSlider
            nodeId={nodeId}
            field={instance as IntegerFieldInputInstance}
            fieldTemplate={template as IntegerFieldInputTemplate}
            settings={settings}
          />
        );
      }
      return (
        <IntegerFieldInput
          nodeId={nodeId}
          field={instance as IntegerFieldInputInstance}
          fieldTemplate={template as IntegerFieldInputTemplate}
          settings={settings?.type === 'integer-field-config' ? settings : undefined}
        />
      );
    }

    if (template.type.name === 'FloatField' && isFloatFieldInputInstance(instance)) {
      if (settings?.type === 'float-field-config' && settings.component === 'slider') {
        return (
          <FloatFieldSlider
            nodeId={nodeId}
            field={instance as FloatFieldInputInstance}
            fieldTemplate={template as FloatFieldInputTemplate}
            settings={settings}
          />
        );
      }
      if (settings?.type === 'float-field-config' && settings.component === 'number-input-and-slider') {
        return (
          <FloatFieldInputAndSlider
            nodeId={nodeId}
            field={instance as FloatFieldInputInstance}
            fieldTemplate={template as FloatFieldInputTemplate}
            settings={settings}
          />
        );
      }
      return (
        <FloatFieldInput
          nodeId={nodeId}
          field={instance as FloatFieldInputInstance}
          fieldTemplate={template as FloatFieldInputTemplate}
          settings={settings?.type === 'float-field-config' ? settings : undefined}
        />
      );
    }

    if (template.type.name === 'BooleanField' && isBooleanFieldInputInstance(instance)) {
      return (
        <BooleanFieldInputComponent
          nodeId={nodeId}
          field={instance as BooleanFieldInputInstance}
          fieldTemplate={template as BooleanFieldInputTemplate}
        />
      );
    }

    if (template.type.name === 'EnumField' && isEnumFieldInputInstance(instance)) {
      return (
        <EnumFieldInputComponent
          nodeId={nodeId}
          field={instance as EnumFieldInputInstance}
          fieldTemplate={template as EnumFieldInputTemplate}
        />
      );
    }

    if (template.type.name === 'BoardField' && isBoardFieldInputInstance(instance)) {
      return (
        <BoardFieldInputComponent
          nodeId={nodeId}
          field={instance as BoardFieldInputInstance}
          fieldTemplate={template as BoardFieldInputTemplate}
        />
      );
    }

    if (template.type.name === 'ModelIdentifierField' && isModelIdentifierFieldInputInstance(instance)) {
      return (
        <ModelIdentifierFieldInputComponent
          nodeId={nodeId}
          field={instance as ModelIdentifierFieldInputInstance}
          fieldTemplate={template as ModelIdentifierFieldInputTemplate}
        />
      );
    }

    if (template.type.name === 'SchedulerField' && isSchedulerFieldInputInstance(instance)) {
      return (
        <SchedulerFieldInputComponent
          nodeId={nodeId}
          field={instance as SchedulerFieldInputInstance}
          fieldTemplate={template as SchedulerFieldInputTemplate}
        />
      );
    }

    if (template.type.name === 'ColorField' && isColorFieldInputInstance(instance)) {
      return (
        <ColorFieldInputComponent
          nodeId={nodeId}
          field={instance as ColorFieldInputInstance}
          fieldTemplate={template as ColorFieldInputTemplate}
        />
      );
    }

    if (template.type.name === 'StylePresetField' && isStylePresetFieldInputInstance(instance)) {
      return (
        <StylePresetFieldInputComponent
          nodeId={nodeId}
          field={instance as StylePresetFieldInputInstance}
          fieldTemplate={template as StylePresetFieldInputTemplate}
        />
      );
    }

    return (
      <Text fontSize="xs" variant="subtext">
        Unsupported dynamic field type: {template.type.name}
      </Text>
    );
  }
);
DynamicFieldInputRenderer.displayName = 'DynamicFieldInputRenderer';

const OutputFields = memo(({ nodeId }: { nodeId: string }) => {
  const fieldNames = useOutputFieldNames();
  return (
    <>
      {fieldNames.map((fieldName, i) => (
        <GridItem gridColumnStart={2} gridRowStart={i + 1} key={`${nodeId}.${fieldName}.output-field`}>
          <OutputFieldGate nodeId={nodeId} fieldName={fieldName}>
            <OutputFieldNodesEditorView nodeId={nodeId} fieldName={fieldName} />
          </OutputFieldGate>
        </GridItem>
      ))}
    </>
  );
});
OutputFields.displayName = 'OutputFields';
