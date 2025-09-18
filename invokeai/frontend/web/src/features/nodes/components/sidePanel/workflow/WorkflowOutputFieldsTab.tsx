import { Button, ButtonGroup, Divider, Flex, Spacer, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { InvocationNodeContextProvider } from 'features/nodes/components/flow/nodes/Invocation/context';
import { NodeFieldElementOverlay } from 'features/nodes/components/sidePanel/builder/NodeFieldElementEditMode';
import {
  $isInPublishFlow,
  $isSelectingOutputNode,
  $outputNodeId,
} from 'features/nodes/components/sidePanel/workflow/publish';
import { useMouseOverFormField } from 'features/nodes/hooks/useMouseOverNode';
import { useNodeTemplateTitleOrThrow } from 'features/nodes/hooks/useNodeTemplateTitleOrThrow';
import { useNodeUserTitleOrThrow } from 'features/nodes/hooks/useNodeUserTitleOrThrow';
import { useOutputFieldTemplate } from 'features/nodes/hooks/useOutputFieldTemplate';
import { useZoomToNode } from 'features/nodes/hooks/useZoomToNode';
import { $templates, workflowOutputFieldsChanged } from 'features/nodes/store/nodesSlice';
import { selectNodes, selectNodesSlice, selectWorkflowOutputFields } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import { type AnyNode, isInvocationNode } from 'features/nodes/types/invocation';
import type { WorkflowOutputField } from 'features/nodes/types/workflow';
import { toast } from 'features/toast/toast';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowLineRightBold, PiTrashBold } from 'react-icons/pi';

type OutputDetail = { nodeId: string; fieldName: string; userLabel: string | null };

const buildOutputDetails = (outputs: WorkflowOutputField[], templates: Templates, nodes: AnyNode[]): OutputDetail[] => {
  const details: OutputDetail[] = [];

  for (const output of outputs) {
    if (details.length >= 1) {
      break;
    }

    const node = nodes.find((n) => n.id === output.nodeId);
    if (!isInvocationNode(node)) {
      continue;
    }

    const template = templates[node.data.type];
    const fieldTemplate = template?.outputs?.[output.fieldName];

    if (!fieldTemplate || fieldTemplate.type.name !== 'ImageField' || fieldTemplate.type.cardinality === 'COLLECTION') {
      continue;
    }

    details.push({
      nodeId: output.nodeId,
      fieldName: output.fieldName,
      userLabel: output.userLabel ?? fieldTemplate.title ?? null,
    });
  }

  return details;
};

const WorkflowOutputFieldsTab = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const store = useAppStore();
  const outputFields = useAppSelector(selectWorkflowOutputFields);
  const nodes = useAppSelector(selectNodes);
  const templates = useStore($templates);
  const outputNodeId = useStore($outputNodeId);
  const isSelectingOutputNode = useStore($isSelectingOutputNode);
  const isInPublishFlow = useStore($isInPublishFlow);
  const selectionSourceRef = useRef<'workflow-output-tab' | null>(null);
  const previousOutputsRef = useRef<WorkflowOutputField[]>(outputFields);

  const outputDetails = useMemo(
    () => buildOutputDetails(outputFields, templates, nodes),
    [outputFields, templates, nodes]
  );

  const revertSelection = useCallback(
    (outputsToRestore: WorkflowOutputField[]) => {
      dispatch(
        workflowOutputFieldsChanged(
          outputsToRestore.map(({ nodeId, fieldName, userLabel }) => ({
            nodeId,
            fieldName,
            userLabel,
          }))
        )
      );
      const restoredNodeId = outputsToRestore[0]?.nodeId ?? null;
      $outputNodeId.set(restoredNodeId);
    },
    [dispatch]
  );

  useEffect(() => {
    previousOutputsRef.current = outputFields;
  }, [outputFields]);

  useEffect(() => {
    if (isInPublishFlow || isSelectingOutputNode) {
      return;
    }
    const currentNodeId = outputFields[0]?.nodeId ?? null;
    if ($outputNodeId.get() !== currentNodeId) {
      $outputNodeId.set(currentNodeId);
    }
  }, [isInPublishFlow, isSelectingOutputNode, outputFields]);

  useEffect(() => {
    if (selectionSourceRef.current !== 'workflow-output-tab') {
      return;
    }
    if (isSelectingOutputNode) {
      return;
    }
    const selectedNodeId = outputNodeId;
    selectionSourceRef.current = null;

    if (!selectedNodeId) {
      return;
    }

    const state = store.getState();
    const nodesState = selectNodesSlice(state);
    const node = nodesState.nodes.find((n) => n.id === selectedNodeId);

    if (!isInvocationNode(node)) {
      toast({
        status: 'error',
        title: t('workflows.builder.outputFieldSelectInvalid', { defaultValue: 'Please select an invocation node.' }),
      });
      revertSelection(previousOutputsRef.current);
      return;
    }

    const template = templates[node.data.type];
    if (!template) {
      toast({
        status: 'error',
        title: t('workflows.builder.outputFieldMissingTemplate', {
          defaultValue: 'Unable to select outputs for this node.',
        }),
      });
      revertSelection(previousOutputsRef.current);
      return;
    }

    const imageOutputEntry = Object.entries(template.outputs).find(([, output]) => {
      return output.type.name === 'ImageField' && output.type.cardinality !== 'COLLECTION';
    });

    if (!imageOutputEntry) {
      toast({
        status: 'error',
        title: t('workflows.builder.outputFieldMustBeImage', {
          defaultValue: 'Selected node must have an image output.',
        }),
      });
      revertSelection(previousOutputsRef.current);
      return;
    }

    const [fieldName, fieldTemplate] = imageOutputEntry;
    dispatch(
      workflowOutputFieldsChanged([
        {
          nodeId: selectedNodeId,
          fieldName,
          userLabel: fieldTemplate.title ?? null,
        },
      ])
    );
  }, [dispatch, isSelectingOutputNode, outputNodeId, revertSelection, store, templates, t]);

  const handleSelectNodeClick = useCallback(() => {
    selectionSourceRef.current = 'workflow-output-tab';
    previousOutputsRef.current = outputFields;
    $outputNodeId.set(null);
    $isSelectingOutputNode.set(true);
  }, [outputFields]);

  const handleClear = useCallback(() => {
    previousOutputsRef.current = [];
    dispatch(workflowOutputFieldsChanged([]));
    $outputNodeId.set(null);
  }, [dispatch]);

  return (
    <Flex flexDir="column" gap={2} h="full">
      <Flex alignItems="center">
        <Text fontWeight="semibold">{t('workflows.builder.outputFieldsTab', 'Output Fields')}</Text>
        <Spacer />
        <ButtonGroup size="sm" variant="ghost" isAttached={false}>
          {outputDetails.length > 0 && (
            <Button leftIcon={<PiTrashBold />} onClick={handleClear}>
              {t('common.clear')}
            </Button>
          )}
          <Button
            leftIcon={<PiArrowLineRightBold />}
            onClick={handleSelectNodeClick}
            isDisabled={isSelectingOutputNode}
          >
            {isSelectingOutputNode
              ? t('workflows.builder.selectingOutputNode')
              : outputDetails.length > 0
                ? t('workflows.builder.changeOutputNode')
                : t('workflows.builder.selectOutputNode')}
          </Button>
        </ButtonGroup>
      </Flex>
      <Divider />
      {outputDetails.length === 0 ? (
        <Text color="warning.300" fontWeight="semibold">
          {outputFields.length === 0
            ? t('workflows.builder.noOutputNodeSelected')
            : t('workflows.builder.outputFieldPending', {
                defaultValue: 'Selected output is unavailable. Check that the node still exists.',
              })}
        </Text>
      ) : (
        outputDetails.map((detail) => (
          <InvocationNodeContextProvider nodeId={detail.nodeId} key={`${detail.nodeId}-${detail.fieldName}`}>
            <SelectedOutputPreview
              nodeId={detail.nodeId}
              fieldName={detail.fieldName}
              fallbackLabel={detail.userLabel ?? detail.fieldName}
            />
          </InvocationNodeContextProvider>
        ))
      )}
    </Flex>
  );
};

export default WorkflowOutputFieldsTab;

const SelectedOutputPreview = ({
  nodeId,
  fieldName,
  fallbackLabel,
}: {
  nodeId: string;
  fieldName: string;
  fallbackLabel: string;
}) => {
  const mouseOverFormField = useMouseOverFormField(nodeId);
  const nodeUserTitle = useNodeUserTitleOrThrow();
  const nodeTemplateTitle = useNodeTemplateTitleOrThrow();
  const fieldTemplate = useOutputFieldTemplate(fieldName);
  const zoomToNode = useZoomToNode(nodeId);

  return (
    <Flex
      flexDir="column"
      position="relative"
      p={2}
      borderRadius="base"
      borderWidth={1}
      gap={1}
      onMouseOver={mouseOverFormField.handleMouseOver}
      onMouseOut={mouseOverFormField.handleMouseOut}
      onClick={zoomToNode}
    >
      <Text fontWeight="semibold">{`${nodeUserTitle || nodeTemplateTitle} -> ${fieldTemplate?.title ?? fallbackLabel}`}</Text>
      <Text variant="subtext">{`${nodeId} -> ${fieldName}`}</Text>
      <NodeFieldElementOverlay nodeId={nodeId} />
    </Flex>
  );
};
