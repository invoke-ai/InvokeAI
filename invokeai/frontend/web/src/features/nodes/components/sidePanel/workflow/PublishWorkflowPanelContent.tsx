import type { ButtonProps } from '@invoke-ai/ui-library';
import {
  Button,
  ButtonGroup,
  Divider,
  Flex,
  ListItem,
  Spacer,
  Text,
  Tooltip,
  UnorderedList,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { $projectUrl } from 'app/store/nanostores/projectId';
import { useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { withResultAsync } from 'common/util/result';
import { parseify } from 'common/util/serialize';
import { ExternalLink } from 'features/gallery/components/ImageViewer/NoContentForViewer';
import { NodeFieldElementOverlay } from 'features/nodes/components/sidePanel/builder/NodeFieldElementEditMode';
import { useDoesWorkflowHaveUnsavedChanges } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import {
  $isInPublishFlow,
  $isReadyToDoValidationRun,
  $isSelectingOutputNode,
  $outputNodeId,
  $validationRunData,
  selectHasUnpublishableNodes,
  usePublishInputs,
} from 'features/nodes/components/sidePanel/workflow/publish';
import { useInputFieldTemplateTitleOrThrow } from 'features/nodes/hooks/useInputFieldTemplateTitleOrThrow';
import { useInputFieldUserTitleOrThrow } from 'features/nodes/hooks/useInputFieldUserTitleOrThrow';
import { useMouseOverFormField } from 'features/nodes/hooks/useMouseOverNode';
import { useNodeTemplateTitleOrThrow } from 'features/nodes/hooks/useNodeTemplateTitleOrThrow';
import { useNodeUserTitleOrThrow } from 'features/nodes/hooks/useNodeUserTitleOrThrow';
import { useOutputFieldNames } from 'features/nodes/hooks/useOutputFieldNames';
import { useOutputFieldTemplate } from 'features/nodes/hooks/useOutputFieldTemplate';
import { useZoomToNode } from 'features/nodes/hooks/useZoomToNode';
import { useEnqueueWorkflows } from 'features/queue/hooks/useEnqueueWorkflows';
import { $isReadyToEnqueue } from 'features/queue/store/readiness';
import { selectAllowPublishWorkflows } from 'features/system/store/configSlice';
import { toast } from 'features/toast/toast';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiArrowLineRightBold, PiLightningFill, PiXBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';
import { assert } from 'tsafe';

const log = logger('generation');

export const PublishWorkflowPanelContent = memo(() => {
  return (
    <Flex flexDir="column" gap={2} h="full">
      <ButtonGroup isAttached={false} size="sm" variant="ghost">
        <Spacer />
        <CancelPublishButton />
        <PublishWorkflowButton />
      </ButtonGroup>
      <ScrollableContent>
        <Flex flexDir="column" gap={2} w="full" h="full">
          <OutputFields />
          <PublishableInputFields />
          <UnpublishableInputFields />
        </Flex>
      </ScrollableContent>
    </Flex>
  );
});
PublishWorkflowPanelContent.displayName = 'PublishWorkflowPanelContent';

const OutputFields = memo(() => {
  const { t } = useTranslation();
  const outputNodeId = useStore($outputNodeId);

  return (
    <Flex flexDir="column" borderWidth={1} borderRadius="base" gap={2} p={2}>
      <Flex alignItems="center">
        <Text fontWeight="semibold">{t('workflows.builder.publishedWorkflowOutputs')}</Text>
        <Spacer />
        <SelectOutputNodeButton variant="link" size="sm" />
      </Flex>

      <Divider />
      {!outputNodeId && (
        <Text fontWeight="semibold" color="error.300">
          {t('workflows.builder.noOutputNodeSelected')}
        </Text>
      )}
      {outputNodeId && <OutputFieldsContent outputNodeId={outputNodeId} />}
    </Flex>
  );
});
OutputFields.displayName = 'OutputFields';

const OutputFieldsContent = memo(({ outputNodeId }: { outputNodeId: string }) => {
  const outputFieldNames = useOutputFieldNames(outputNodeId);

  return (
    <>
      {outputFieldNames.map((fieldName) => (
        <NodeOutputFieldPreview key={`${outputNodeId}-${fieldName}`} nodeId={outputNodeId} fieldName={fieldName} />
      ))}
    </>
  );
});
OutputFieldsContent.displayName = 'OutputFieldsContent';

const PublishableInputFields = memo(() => {
  const { t } = useTranslation();
  const inputs = usePublishInputs();

  if (inputs.publishable.length === 0) {
    return (
      <Flex flexDir="column" borderWidth={1} borderRadius="base" gap={2} p={2}>
        <Text fontWeight="semibold" color="warning.300">
          {t('workflows.builder.noPublishableInputs')}
        </Text>
      </Flex>
    );
  }

  return (
    <Flex flexDir="column" borderWidth={1} borderRadius="base" gap={2} p={2}>
      <Text fontWeight="semibold">{t('workflows.builder.publishedWorkflowInputs')}</Text>
      <Divider />
      {inputs.publishable.map(({ nodeId, fieldName }) => {
        return <NodeInputFieldPreview key={`${nodeId}-${fieldName}`} nodeId={nodeId} fieldName={fieldName} />;
      })}
    </Flex>
  );
});
PublishableInputFields.displayName = 'PublishableInputFields';

const UnpublishableInputFields = memo(() => {
  const { t } = useTranslation();
  const inputs = usePublishInputs();

  if (inputs.unpublishable.length === 0) {
    return null;
  }

  return (
    <Flex flexDir="column" borderWidth={1} borderRadius="base" gap={2} p={2}>
      <Text fontWeight="semibold" color="warning.300">
        {t('workflows.builder.unpublishableInputs')}
      </Text>
      <Divider />
      {inputs.unpublishable.map(({ nodeId, fieldName }) => {
        return <NodeInputFieldPreview key={`${nodeId}-${fieldName}`} nodeId={nodeId} fieldName={fieldName} />;
      })}
    </Flex>
  );
});
UnpublishableInputFields.displayName = 'UnpublishableInputFields';

const SelectOutputNodeButton = memo((props: ButtonProps) => {
  const { t } = useTranslation();
  const outputNodeId = useStore($outputNodeId);
  const isSelectingOutputNode = useStore($isSelectingOutputNode);
  const onClick = useCallback(() => {
    $outputNodeId.set(null);
    $isSelectingOutputNode.set(true);
  }, []);
  return (
    <Button
      leftIcon={<PiArrowLineRightBold />}
      isDisabled={isSelectingOutputNode}
      tooltip={isSelectingOutputNode ? t('workflows.builder.selectingOutputNodeDesc') : undefined}
      onClick={onClick}
      {...props}
    >
      {isSelectingOutputNode
        ? t('workflows.builder.selectingOutputNode')
        : outputNodeId
          ? t('workflows.builder.changeOutputNode')
          : t('workflows.builder.selectOutputNode')}
    </Button>
  );
});
SelectOutputNodeButton.displayName = 'SelectOutputNodeButton';

const CancelPublishButton = memo(() => {
  const { t } = useTranslation();
  const onClick = useCallback(() => {
    $isInPublishFlow.set(false);
    $isSelectingOutputNode.set(false);
    $outputNodeId.set(null);
  }, []);
  return (
    <Button leftIcon={<PiXBold />} onClick={onClick}>
      {t('common.cancel')}
    </Button>
  );
});
CancelPublishButton.displayName = 'CancelDeployButton';

const PublishWorkflowButton = memo(() => {
  const { t } = useTranslation();
  const isReadyToDoValidationRun = useStore($isReadyToDoValidationRun);
  const isReadyToEnqueue = useStore($isReadyToEnqueue);
  const doesWorkflowHaveUnsavedChanges = useDoesWorkflowHaveUnsavedChanges();
  const hasUnpublishableNodes = useAppSelector(selectHasUnpublishableNodes);
  const outputNodeId = useStore($outputNodeId);
  const isSelectingOutputNode = useStore($isSelectingOutputNode);
  const inputs = usePublishInputs();
  const allowPublishWorkflows = useAppSelector(selectAllowPublishWorkflows);

  const projectUrl = useStore($projectUrl);

  const enqueue = useEnqueueWorkflows();
  const onClick = useCallback(async () => {
    const result = await withResultAsync(() => enqueue(true, true));
    if (result.isErr()) {
      toast({
        id: 'TOAST_PUBLISH_FAILED',
        status: 'error',
        title: t('workflows.builder.publishFailed'),
        description: t('workflows.builder.publishFailedDesc'),
        duration: null,
      });
      log.error({ error: serializeError(result.error) }, 'Failed to enqueue batch');
    } else {
      toast({
        id: 'TOAST_PUBLISH_SUCCESSFUL',
        status: 'success',
        title: t('workflows.builder.publishSuccess'),
        description: (
          <Trans
            i18nKey="workflows.builder.publishSuccessDesc"
            components={{
              LinkComponent: <ExternalLink href={projectUrl ?? ''} />,
            }}
          />
        ),
        duration: null,
      });
      assert(result.value.enqueueResult.batch.batch_id);
      assert(result.value.batchConfig.validation_run_data);
      $validationRunData.set({
        batchId: result.value.enqueueResult.batch.batch_id,
        workflowId: result.value.batchConfig.validation_run_data.workflow_id,
      });
      log.debug(parseify(result.value), 'Enqueued batch');
    }
  }, [enqueue, projectUrl, t]);

  return (
    <PublishTooltip
      isWorkflowSaved={!doesWorkflowHaveUnsavedChanges}
      hasUnpublishableNodes={hasUnpublishableNodes}
      isReadyToEnqueue={isReadyToEnqueue}
      hasOutputNode={outputNodeId !== null && !isSelectingOutputNode}
      hasPublishableInputs={inputs.publishable.length > 0}
      hasUnpublishableInputs={inputs.unpublishable.length > 0}
    >
      <Button
        leftIcon={<PiLightningFill />}
        isDisabled={
          !allowPublishWorkflows ||
          !isReadyToEnqueue ||
          doesWorkflowHaveUnsavedChanges ||
          hasUnpublishableNodes ||
          !isReadyToDoValidationRun ||
          !(outputNodeId !== null && !isSelectingOutputNode)
        }
        onClick={onClick}
      >
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
  const allowPublishWorkflows = useAppSelector(selectAllowPublishWorkflows);
  const isReadyToEnqueue = useStore($isReadyToEnqueue);
  const doesWorkflowHaveUnsavedChanges = useDoesWorkflowHaveUnsavedChanges();
  const hasUnpublishableNodes = useAppSelector(selectHasUnpublishableNodes);
  const inputs = usePublishInputs();

  const onClick = useCallback(() => {
    $isInPublishFlow.set(true);
  }, []);

  return (
    <PublishTooltip
      isWorkflowSaved={!doesWorkflowHaveUnsavedChanges}
      hasUnpublishableNodes={hasUnpublishableNodes}
      isReadyToEnqueue={isReadyToEnqueue}
      hasOutputNode={true}
      hasPublishableInputs={inputs.publishable.length > 0}
      hasUnpublishableInputs={inputs.unpublishable.length > 0}
    >
      <Button
        onClick={onClick}
        leftIcon={<PiLightningFill />}
        variant="ghost"
        size="sm"
        isDisabled={
          !allowPublishWorkflows || !isReadyToEnqueue || doesWorkflowHaveUnsavedChanges || hasUnpublishableNodes
        }
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
    hasUnpublishableNodes,
    isReadyToEnqueue,
    hasOutputNode,
    hasPublishableInputs,
    hasUnpublishableInputs,
    children,
  }: PropsWithChildren<{
    isWorkflowSaved: boolean;
    hasUnpublishableNodes: boolean;
    isReadyToEnqueue: boolean;
    hasOutputNode: boolean;
    hasPublishableInputs: boolean;
    hasUnpublishableInputs: boolean;
  }>) => {
    const { t } = useTranslation();
    const warnings = useMemo(() => {
      const _warnings: string[] = [];
      if (!hasPublishableInputs) {
        _warnings.push(t('workflows.builder.warningWorkflowHasNoPublishableInputFields'));
      }
      if (hasUnpublishableInputs) {
        _warnings.push(t('workflows.builder.warningWorkflowHasUnpublishableInputFields'));
      }
      return _warnings;
    }, [hasPublishableInputs, hasUnpublishableInputs, t]);
    const errors = useMemo(() => {
      const _errors: string[] = [];
      if (!isWorkflowSaved) {
        _errors.push(t('workflows.builder.errorWorkflowHasUnsavedChanges'));
      }
      if (hasUnpublishableNodes) {
        _errors.push(t('workflows.builder.errorWorkflowHasUnpublishableNodes'));
      }
      if (!isReadyToEnqueue) {
        _errors.push(t('workflows.builder.errorWorkflowHasInvalidGraph'));
      }
      if (!hasOutputNode) {
        _errors.push(t('workflows.builder.errorWorkflowHasNoOutputNode'));
      }
      return _errors;
    }, [hasUnpublishableNodes, hasOutputNode, isReadyToEnqueue, isWorkflowSaved, t]);

    if (errors.length === 0 && warnings.length === 0) {
      return children;
    }

    return (
      <Tooltip
        label={
          <Flex flexDir="column">
            {errors.length > 0 && (
              <>
                <Text color="error.700" fontWeight="semibold">
                  {t('workflows.builder.cannotPublish')}:
                </Text>
                <UnorderedList>
                  {errors.map((problem, index) => (
                    <ListItem key={index}>{problem}</ListItem>
                  ))}
                </UnorderedList>
              </>
            )}
            {warnings.length > 0 && (
              <>
                <Text color="warning.700" fontWeight="semibold">
                  {t('workflows.builder.publishWarnings')}:
                </Text>
                <UnorderedList>
                  {warnings.map((problem, index) => (
                    <ListItem key={index}>{problem}</ListItem>
                  ))}
                </UnorderedList>
              </>
            )}
          </Flex>
        }
      >
        {children}
      </Tooltip>
    );
  }
);
PublishTooltip.displayName = 'PublishTooltip';
