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
import { useInvoke } from 'features/queue/hooks/useInvoke';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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
  const { t } = useTranslation();
  const isSelectingOutputNode = useStore($isSelectingOutputNode);
  const onClick = useCallback(() => {
    $outputNodeId.set(null);
    $isSelectingOutputNode.set(true);
  }, []);
  return (
    <Button isDisabled={isSelectingOutputNode} onClick={onClick}>
      {t('workflows.builder.selectOutputNode')}
    </Button>
  );
});
SelectOutputNodeButton.displayName = 'SelectOutputNodeButton';

const CancelDeployButton = memo(() => {
  const { t } = useTranslation();
  const onClick = useCallback(() => {
    $isInDeployFlow.set(false);
    $isSelectingOutputNode.set(false);
    $outputNodeId.set(null);
  }, []);
  return <Button onClick={onClick}>{t('common.cancel')}</Button>;
});
CancelDeployButton.displayName = 'CancelDeployButton';

const DoValidationRunButton = memo(() => {
  const { t } = useTranslation();
  const isReadyToDoValidationRun = useStore($isReadyToDoValidationRun);
  const invoke = useInvoke();
  const onClick = useCallback(() => {
    invoke.enqueue(true, true);
  }, [invoke]);
  return (
    <Button isDisabled={!isReadyToDoValidationRun || invoke.isDisabled} onClick={onClick}>
      {t('workflows.builder.publish')}
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
