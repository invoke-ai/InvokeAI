import { Button, ButtonGroup, Flex, ListItem, Text, Tooltip, UnorderedList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { parseify } from 'common/util/serialize';
import {
  $isInDeployFlow,
  $isReadyToDoValidationRun,
  $isSelectingOutputNode,
  $outputNodeId,
  resetPublishState,
} from 'features/nodes/components/sidePanel/builder/deploy';
import { NodeFieldElementOverlay } from 'features/nodes/components/sidePanel/builder/NodeFieldElementEditMode';
import { useInputFieldTemplateTitleOrThrow } from 'features/nodes/hooks/useInputFieldTemplateTitleOrThrow';
import { useInputFieldUserTitleOrThrow } from 'features/nodes/hooks/useInputFieldUserTitleOrThrow';
import { useMouseOverFormField } from 'features/nodes/hooks/useMouseOverNode';
import { useNodeTemplateTitleOrThrow } from 'features/nodes/hooks/useNodeTemplateTitleOrThrow';
import { useNodeUserTitleOrThrow } from 'features/nodes/hooks/useNodeUserTitleOrThrow';
import { useOutputFieldNames } from 'features/nodes/hooks/useOutputFieldNames';
import { useOutputFieldTemplate } from 'features/nodes/hooks/useOutputFieldTemplate';
import { useZoomToNode } from 'features/nodes/hooks/useZoomToNode';
import { selectHasBatchOrGeneratorNodes } from 'features/nodes/store/selectors';
import { selectIsWorkflowSaved, selectNodeFieldElementsDeduped } from 'features/nodes/store/workflowSlice';
import { useEnqueueWorkflows } from 'features/queue/hooks/useEnqueueWorkflows';
import { $isReadyToEnqueue } from 'features/queue/store/readiness';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { toast } from 'features/toast/toast';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightningFill } from 'react-icons/pi';
import { serializeError } from 'serialize-error';

const log = logger('generation');

export const PublishWorkflowPanelContent = memo(() => {
  const nodeFieldElements = useAppSelector(selectNodeFieldElementsDeduped);
  const outputNodeId = useStore($outputNodeId);
  return (
    <Flex flexDir="column">
      <ButtonGroup isAttached={false}>
        <SelectOutputNodeButton />
        <PublishWorkflowButton />
        <CancelPublishButton />
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
PublishWorkflowPanelContent.displayName = 'DeployWorkflowPanelContent';

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

const CancelPublishButton = memo(() => {
  const { t } = useTranslation();
  const onClick = useCallback(() => {
    $isInDeployFlow.set(false);
    $isSelectingOutputNode.set(false);
    $outputNodeId.set(null);
  }, []);
  return <Button onClick={onClick}>{t('common.cancel')}</Button>;
});
CancelPublishButton.displayName = 'CancelDeployButton';

const PublishWorkflowButton = memo(() => {
  const { t } = useTranslation();
  const isReadyToDoValidationRun = useStore($isReadyToDoValidationRun);
  const isReadyToEnqueue = useStore($isReadyToEnqueue);
  const isWorkflowSaved = useAppSelector(selectIsWorkflowSaved);
  const hasBatchOrGeneratorNodes = useAppSelector(selectHasBatchOrGeneratorNodes);
  const outputNodeId = useStore($outputNodeId);
  const isSelectingOutputNode = useStore($isSelectingOutputNode);

  const enqueue = useEnqueueWorkflows();
  const onClick = useCallback(async () => {
    const result = await withResultAsync(() => enqueue(true, true));
    if (result.isErr()) {
      toast({
        status: 'error',
        title: t('workflows.builder.publishFailed'),
        description: t('workflows.builder.publishFailedDesc'),
      });
      log.error({ error: serializeError(result.error) }, 'Failed to enqueue batch');
    } else {
      toast({
        status: 'success',
        title: t('workflows.builder.publishSuccess'),
        description: t('workflows.builder.publishSuccessDesc'),
      });
      resetPublishState();
      log.debug(parseify(result.value), 'Enqueued batch');
    }
  }, [enqueue, t]);

  return (
    <PublishTooltip
      isWorkflowSaved={isWorkflowSaved}
      hasBatchOrGeneratorNodes={hasBatchOrGeneratorNodes}
      isReadyToEnqueue={isReadyToEnqueue}
      hasOutputNode={outputNodeId !== null && !isSelectingOutputNode}
    >
      <Button isDisabled={!isReadyToDoValidationRun || !isReadyToEnqueue} onClick={onClick}>
        {t('workflows.builder.publish')}
      </Button>
    </PublishTooltip>
  );
});
PublishWorkflowButton.displayName = 'DoValidationRunButton';

const NodeInputFieldPreview = memo(({ nodeId, fieldName }: { nodeId: string; fieldName: string }) => {
  const mouseOverFormField = useMouseOverFormField(nodeId);
  const nodeUserTitle = useNodeUserTitleOrThrow(nodeId);
  const nodeTemplateTitle = useNodeTemplateTitleOrThrow(nodeId);
  const fieldUserTitle = useInputFieldUserTitleOrThrow(nodeId, fieldName);
  const fieldTemplateTitle = useInputFieldTemplateTitleOrThrow(nodeId, fieldName);
  const zoomToNode = useZoomToNode(nodeId);

  return (
    <Flex
      flexDir="column"
      position="relative"
      p={2}
      borderRadius="base"
      onMouseOver={mouseOverFormField.handleMouseOver}
      onMouseOut={mouseOverFormField.handleMouseOut}
      onClick={zoomToNode}
    >
      <Text fontWeight="semibold">{`${nodeUserTitle || nodeTemplateTitle} -> ${fieldUserTitle || fieldTemplateTitle}`}</Text>
      <Text variant="subtext">{`${nodeId} -> ${fieldName}`}</Text>
      <NodeFieldElementOverlay nodeId={nodeId} />
    </Flex>
  );
});
NodeInputFieldPreview.displayName = 'NodeInputFieldPreview';

const NodeOutputFieldPreview = memo(({ nodeId, fieldName }: { nodeId: string; fieldName: string }) => {
  const mouseOverFormField = useMouseOverFormField(nodeId);
  const nodeUserTitle = useNodeUserTitleOrThrow(nodeId);
  const nodeTemplateTitle = useNodeTemplateTitleOrThrow(nodeId);
  const fieldTemplate = useOutputFieldTemplate(nodeId, fieldName);
  const zoomToNode = useZoomToNode(nodeId);

  return (
    <Flex
      flexDir="column"
      position="relative"
      p={2}
      borderRadius="base"
      onMouseOver={mouseOverFormField.handleMouseOver}
      onMouseOut={mouseOverFormField.handleMouseOut}
      onClick={zoomToNode}
    >
      <Text fontWeight="semibold">{`${nodeUserTitle || nodeTemplateTitle} -> ${fieldTemplate.title}`}</Text>
      <Text variant="subtext">{`${nodeId} -> ${fieldName}`}</Text>
      <NodeFieldElementOverlay nodeId={nodeId} />
    </Flex>
  );
});
NodeOutputFieldPreview.displayName = 'NodeOutputFieldPreview';

export const StartPublishFlowButton = memo(() => {
  const { t } = useTranslation();
  const deployWorkflowIsEnabled = useFeatureStatus('deployWorkflow');
  const isReadyToEnqueue = useStore($isReadyToEnqueue);
  const isWorkflowSaved = useAppSelector(selectIsWorkflowSaved);
  const hasBatchOrGeneratorNodes = useAppSelector(selectHasBatchOrGeneratorNodes);

  const onClick = useCallback(() => {
    $isInDeployFlow.set(true);
  }, []);

  return (
    <PublishTooltip
      isWorkflowSaved={isWorkflowSaved}
      hasBatchOrGeneratorNodes={hasBatchOrGeneratorNodes}
      isReadyToEnqueue={isReadyToEnqueue}
      hasOutputNode={true}
    >
      <Button
        onClick={onClick}
        leftIcon={<PiLightningFill />}
        variant="ghost"
        size="sm"
        isDisabled={!deployWorkflowIsEnabled || !isWorkflowSaved || hasBatchOrGeneratorNodes}
      >
        {t('workflows.builder.publish')}
      </Button>
    </PublishTooltip>
  );
});

StartPublishFlowButton.displayName = 'StartPublishFlowButton';

const PublishTooltip = memo(
  ({
    isWorkflowSaved,
    hasBatchOrGeneratorNodes,
    isReadyToEnqueue,
    hasOutputNode,
    children,
  }: PropsWithChildren<{
    isWorkflowSaved: boolean;
    hasBatchOrGeneratorNodes: boolean;
    isReadyToEnqueue: boolean;
    hasOutputNode: boolean;
  }>) => {
    const { t } = useTranslation();
    const problems = useMemo(() => {
      const _problems: string[] = [];
      if (!isWorkflowSaved) {
        _problems.push(t('workflows.builder.cannotPublishUnsavedWorkflow'));
      }
      if (hasBatchOrGeneratorNodes) {
        _problems.push(t('workflows.builder.cannotPublishWorkflowWithBatchOrGeneratorNodes'));
      }
      if (!isReadyToEnqueue) {
        _problems.push(t('workflows.builder.cannotPublishInvalidWorkflow'));
      }
      if (!hasOutputNode) {
        _problems.push(t('workflows.builder.cannotPublishWorkflowWithoutOutputNode'));
      }
      return _problems;
    }, [hasBatchOrGeneratorNodes, hasOutputNode, isReadyToEnqueue, isWorkflowSaved, t]);

    if (problems.length === 0) {
      return children;
      // return t('workflows.builder.publish');
    }

    return (
      <Tooltip
        label={
          <Flex flexDir="column">
            <Text>{t('workflows.builder.cannotPublish')}:</Text>
            <UnorderedList>
              {problems.map((problem, index) => (
                <ListItem key={index}>{problem}</ListItem>
              ))}
            </UnorderedList>
          </Flex>
        }
      >
        {children}
      </Tooltip>
    );
  }
);
PublishTooltip.displayName = 'PublishTooltip';
