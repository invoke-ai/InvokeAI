import { Button, ButtonGroup, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import {
  $isInDeployFlow,
  $isReadyToDoValidationRun,
  $isSelectingOutputNode,
  $outputNodeId,
} from 'features/nodes/components/sidePanel/builder/deploy';
import { useInputFieldTemplateTitleOrThrow } from 'features/nodes/hooks/useInputFieldTemplateTitleOrThrow';
import { useInputFieldUserTitleOrThrow } from 'features/nodes/hooks/useInputFieldUserTitleOrThrow';
import { useNodeTemplateTitleOrThrow } from 'features/nodes/hooks/useNodeTemplateTitleOrThrow';
import { useNodeUserTitleOrThrow } from 'features/nodes/hooks/useNodeUserTitleOrThrow';
import { useOutputFieldNames } from 'features/nodes/hooks/useOutputFieldNames';
import { useOutputFieldTemplate } from 'features/nodes/hooks/useOutputFieldTemplate';
import { selectNodeFieldElementsDeduped } from 'features/nodes/store/workflowSlice';
import { memo, useCallback } from 'react';

export const DeployWorkflowPanelContent = memo(() => {
  const nodeFieldElements = useAppSelector(selectNodeFieldElementsDeduped);
  const outputNodeId = useStore($outputNodeId);
  return (
    <Flex flexDir="column">
      <ButtonGroup isAttached={false}>
        <SelectOutputNodeButton />
        <DoValidationRunButton />
        <CancelDeployButton />
      </ButtonGroup>
      {outputNodeId !== null && <OutputNode outputNodeId={outputNodeId} />}
      <Flex flexDir="column" borderWidth={1}>
        {nodeFieldElements.length !== 0 && <Text fontWeight="semibold">Input Fields</Text>}
        {nodeFieldElements.map((el) => {
          const { nodeId, fieldName } = el.data.fieldIdentifier;
          return <NodeInputFieldPreview key={`${nodeId}-${fieldName}`} nodeId={nodeId} fieldName={fieldName} />;
        })}
      </Flex>
    </Flex>
  );
});
DeployWorkflowPanelContent.displayName = 'DeployWorkflowPanelContent';

const OutputNode = memo(({ outputNodeId }: { outputNodeId: string }) => {
  const resetOutputNode = useCallback(() => {
    $outputNodeId.set(null);
  }, []);
  const nodeUserTitle = useNodeUserTitleOrThrow(outputNodeId);
  const nodeTemplateTitle = useNodeTemplateTitleOrThrow(outputNodeId);
  const outputFieldNames = useOutputFieldNames(outputNodeId);

  return (
    <Flex flexDir="column" borderWidth={1}>
      <Text fontWeight="semibold">Output fields on selected output node</Text>
      {outputFieldNames.map((fieldName) => (
        <NodeOutputFieldPreview key={`${outputNodeId}-${fieldName}`} nodeId={outputNodeId} fieldName={fieldName} />
      ))}
    </Flex>
  );
});
OutputNode.displayName = 'OutputNode';

const SelectOutputNodeButton = memo(() => {
  const isSelectingOutputNode = useStore($isSelectingOutputNode);
  return (
    <Button
      isDisabled={isSelectingOutputNode}
      onClick={() => {
        $outputNodeId.set(null);
        $isSelectingOutputNode.set(true);
      }}
    >
      Select output node
    </Button>
  );
});
SelectOutputNodeButton.displayName = 'SelectOutputNodeButton';

const CancelDeployButton = memo(() => {
  return (
    <Button
      onClick={() => {
        $isInDeployFlow.set(false);
        $isSelectingOutputNode.set(false);
        $outputNodeId.set(null);
      }}
    >
      Cancel
    </Button>
  );
});
CancelDeployButton.displayName = 'CancelDeployButton';

const DoValidationRunButton = memo(() => {
  const isReadyToDoValidationRun = useStore($isReadyToDoValidationRun);
  return (
    <Button isDisabled={!isReadyToDoValidationRun} onClick={() => {}}>
      Do Validation Run
    </Button>
  );
});
DoValidationRunButton.displayName = 'DoValidationRunButton';

const NodeInputFieldPreview = memo(({ nodeId, fieldName }: { nodeId: string; fieldName: string }) => {
  const nodeUserTitle = useNodeUserTitleOrThrow(nodeId);
  const nodeTemplateTitle = useNodeTemplateTitleOrThrow(nodeId);
  const fieldUserTitle = useInputFieldUserTitleOrThrow(nodeId, fieldName);
  const fieldTemplateTitle = useInputFieldTemplateTitleOrThrow(nodeId, fieldName);
  return (
    <Flex flexDir="column">
      <Text fontWeight="semibold">{`${nodeUserTitle || nodeTemplateTitle} -> ${fieldUserTitle || fieldTemplateTitle}`}</Text>
      <Text variant="subtext">{`${nodeId} -> ${fieldName}`}</Text>
    </Flex>
  );
});
NodeInputFieldPreview.displayName = 'NodeInputFieldPreview';

const NodeOutputFieldPreview = memo(({ nodeId, fieldName }: { nodeId: string; fieldName: string }) => {
  const nodeUserTitle = useNodeUserTitleOrThrow(nodeId);
  const nodeTemplateTitle = useNodeTemplateTitleOrThrow(nodeId);
  const fieldTemplate = useOutputFieldTemplate(nodeId, fieldName);
  return (
    <Flex flexDir="column">
      <Text fontWeight="semibold">{`${nodeUserTitle || nodeTemplateTitle} -> ${fieldTemplate.title}`}</Text>
      <Text variant="subtext">{`${nodeId} -> ${fieldName}`}</Text>
    </Flex>
  );
});
NodeOutputFieldPreview.displayName = 'NodeOutputFieldPreview';
