import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Flex, Grid, GridItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { InputFieldEditModeNodes } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldEditModeNodes';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { OutputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldGate';
import { OutputFieldNodesEditorView } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldNodesEditorView';
import InvocationNodeFooter from 'features/nodes/components/flow/nodes/Invocation/InvocationNodeFooter';
import InvocationNodeHeader from 'features/nodes/components/flow/nodes/Invocation/InvocationNodeHeader';
import { useInputFieldInstance } from 'features/nodes/hooks/useInputFieldInstance';
import { useOutputFieldNames } from 'features/nodes/hooks/useOutputFieldNames';
import { useWithFooter } from 'features/nodes/hooks/useWithFooter';
import { $templates, callSavedWorkflowDynamicFieldsChanged } from 'features/nodes/store/nodesSlice';
import type { SavedWorkflowFieldInputInstance } from 'features/nodes/types/field';
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
  w: 'full',
};

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
          <DynamicFieldRow
            key={field.fieldName}
            nodeId={nodeId}
            fieldName={field.fieldName}
            settings={field.settings}
          />
        ))}
      </>
    );
  }
);
DynamicFieldsSection.displayName = 'DynamicFieldsSection';

const DynamicFieldRow = memo(
  ({
    nodeId,
    fieldName,
    settings,
  }: {
    nodeId: string;
    fieldName: string;
    settings: ReturnType<typeof getSavedWorkflowDynamicFields>[number]['settings'];
  }) => {
    return (
      <Flex sx={dynamicFieldSx}>
        <InputFieldGate nodeId={nodeId} fieldName={fieldName}>
          <InputFieldEditModeNodes nodeId={nodeId} fieldName={fieldName} settings={settings} />
        </InputFieldGate>
      </Flex>
    );
  }
);
DynamicFieldRow.displayName = 'DynamicFieldRow';

const OutputFields = memo(({ nodeId }: { nodeId: string }) => {
  const fieldNames = useOutputFieldNames();

  if (fieldNames.length === 0) {
    return null;
  }

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
