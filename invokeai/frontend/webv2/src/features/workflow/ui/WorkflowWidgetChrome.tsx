import type { InvocationTemplate, XYPosition } from '@features/workflow/contracts';

import { Box, HStack, Icon, Menu, Text } from '@chakra-ui/react';
import { updateLoadedWorkflowNodes } from '@features/workflow/data/templates';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  buildConnectorNode,
  buildCurrentImageNode,
  buildInvocationNode,
  buildNotesNode,
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  createWorkflowId,
  getCompatibleInputTemplate,
  getCompatibleOutputTemplate,
  hasMultipleWorkflowReturnNodes,
  LOOP_LINKAGE_FIELD,
  resolveConnectorSource,
  shouldAddForReturnLoopLinkage,
  parseWorkflowJson,
} from '@features/workflow/utility';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, IconButton, RenameDialog, Tooltip } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import {
  BookmarkIcon,
  CloudAlertIcon,
  CloudCheckIcon,
  CloudIcon,
  LibraryIcon,
  PlusIcon,
  RefreshCwIcon,
  TriangleAlertIcon,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState, type ChangeEvent, type ElementType } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowWidgetLabelProps, WorkflowWidgetViewProps } from './contracts';
import type { WorkflowProjectPersistence } from './WorkflowUiContext';

import { AddNodeDialog } from './editor/AddNodeDialog';
import { CallSavedWorkflowSyncRuntime } from './editor/CallSavedWorkflowSyncRuntime';
import { getWorkflowFlowInstance } from './editor/flowInstanceStore';
import { releaseWorkflowViewportsExcept } from './editor/workflowViewportStore';
import { isUpdatableSource } from './library/projectWorkflowEntries';
import { WorkflowLibraryDialog } from './library/WorkflowLibraryDialog';
import { WorkflowPublicationHost } from './library/WorkflowPublicationHost';
import { PendingWorkflowLoader } from './PendingLibraryWorkflowLoader';
import { copyWorkflowJson, downloadWorkflowJson } from './workflowTransfer';
import {
  useWorkflowHostCommands,
  useWorkflowNotifications,
  useWorkflowPersistenceSelector,
  useWorkflowProjectSelector,
  useWorkflowUi,
} from './WorkflowUiContext';
import {
  openWorkflowLibraryAtProjectWorkflow,
  requestWorkflowImport,
  requestWorkflowRename,
  requestWorkflowPublication,
  setAddNodeOpen,
  setWorkflowLibraryOpen,
  workflowUiStore,
} from './workflowUiStore';

/**
 * Keep dialogs mounted here and drive them through workflowUiStore; manifest slots contribute label, quick
 * actions, and shared menu actions.
 */

/** How the project's own persistence is shown beside the library actions. */
const PERSISTENCE_VIEW: Record<
  WorkflowProjectPersistence['status'],
  { color: string; icon: ElementType; labelKey: string }
> = {
  conflict: { color: 'fg.warning', icon: TriangleAlertIcon, labelKey: 'widgets.workflow.persistenceConflict' },
  error: { color: 'fg.error', icon: CloudAlertIcon, labelKey: 'widgets.workflow.persistenceError' },
  pending: { color: 'fg.subtle', icon: CloudIcon, labelKey: 'widgets.workflow.persistencePending' },
  saved: { color: 'fg.subtle', icon: CloudCheckIcon, labelKey: 'widgets.workflow.persistenceSaved' },
  saving: { color: 'fg.subtle', icon: RefreshCwIcon, labelKey: 'widgets.workflow.persistenceSaving' },
};

export const WorkflowWidgetLabel = ({ region }: WorkflowWidgetLabelProps) => {
  const { t } = useTranslation();
  const workflowName = useWorkflowProjectSelector((project) => project.projectGraph.name);
  const projectId = useWorkflowProjectSelector((project) => project.id);
  const activeWorkflowId = useWorkflowProjectSelector((project) => project.activeWorkflowId);
  const openProjectWorkflows = useCallback(
    () => openWorkflowLibraryAtProjectWorkflow(projectId, activeWorkflowId),
    [activeWorkflowId, projectId]
  );

  if (region !== 'center') {
    return (
      <Text fontSize="xs" fontWeight="700">
        {t('widgets.labels.workflow')}
      </Text>
    );
  }

  // Center chrome already names the widget; the workflow name opens the project's workflows with this one selected.
  const displayName = workflowName || t('widgets.workflow.untitled');

  return (
    <HStack flex="1" gap="1" minW="0">
      <Text color="fg.subtle" flexShrink={0} fontSize="xs">
        /
      </Text>
      <Tooltip content={t('widgets.workflow.projectWorkflows')}>
        <Button
          aria-label={t('widgets.workflow.openProjectWorkflows', { name: displayName })}
          // The button recipe refuses to shrink; the name must give way before the actions island overlaps it.
          flexShrink={1}
          maxW="16rem"
          minW="0"
          overflow="hidden"
          size="2xs"
          variant="ghost"
          onClick={openProjectWorkflows}
        >
          <Icon as={LibraryIcon} boxSize="3.5" flexShrink={0} />
          <MiddleTruncate color={workflowName ? undefined : 'fg.subtle'} fontWeight="600" minW="0" text={displayName} />
        </Button>
      </Tooltip>
    </HStack>
  );
};

/** Entries contributed to the shared widget actions menu. */
export const WorkflowMenuItems = (_props: WorkflowWidgetViewProps) => {
  const { t } = useTranslation();
  const { getProjectGraph } = useWorkflowUi();
  const { widgets } = useWorkflowHostCommands();
  const { createWorkflow } = useProjectGraphCommands();
  const notify = useWorkflowNotifications();
  const activeWorkflowId = useWorkflowProjectSelector((project) => project.activeWorkflowId);
  const source = useWorkflowProjectSelector((project) => project.activeWorkflow.source);
  const canUpdateSource = isUpdatableSource(source);

  const openDetailsPanel = useCallback(() => {
    widgets.open({ region: 'left', widgetId: 'workflow' });
    widgets.patchValues('workflow', { editTab: 'details', panelMode: 'edit' });
  }, [widgets]);
  const exportWorkflow = useCallback(() => {
    const projectGraph = getProjectGraph();

    if (hasMultipleWorkflowReturnNodes(projectGraph)) {
      notify.error(t('projects.exportFailed'), t('workflowLibrary.multipleWorkflowReturnNodesForTransfer'));
      return;
    }

    downloadWorkflowJson(projectGraph);
  }, [getProjectGraph, notify, t]);
  const copyWorkflow = useCallback(() => {
    const projectGraph = getProjectGraph();

    if (hasMultipleWorkflowReturnNodes(projectGraph)) {
      notify.error(t('widgets.workflow.copyJsonFailed'), t('workflowLibrary.multipleWorkflowReturnNodesForTransfer'));
      return;
    }

    copyWorkflowJson(projectGraph)
      .then(() => notify.success(t('widgets.workflow.copyJsonSuccess')))
      .catch(() => notify.error(t('widgets.workflow.copyJsonFailed')));
  }, [getProjectGraph, notify, t]);
  const saveToLibrary = useCallback(
    () => requestWorkflowPublication({ kind: 'save-as-new', workflowId: activeWorkflowId }),
    [activeWorkflowId]
  );
  const updateTemplate = useCallback(
    () => requestWorkflowPublication({ kind: 'update-source', workflowId: activeWorkflowId }),
    [activeWorkflowId]
  );
  const openLibrary = useCallback(() => setWorkflowLibraryOpen(true), []);
  const renameActiveWorkflow = useCallback(() => requestWorkflowRename(activeWorkflowId), [activeWorkflowId]);
  // Nothing is replaced: a new workflow is added beside the others, so no confirmation stands in the way.
  const newWorkflow = useCallback(() => createWorkflow(), [createWorkflow]);

  return (
    <Menu.ItemGroup>
      <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
        {t('widgets.labels.workflow')}
      </Menu.ItemGroupLabel>
      <Menu.Item value="details" onClick={openDetailsPanel}>
        {t('widgets.workflow.detailsWithEllipsis')}
      </Menu.Item>
      <Menu.Item value="rename" onClick={renameActiveWorkflow}>
        {t('widgets.workflow.renameWithEllipsis')}
      </Menu.Item>
      <Menu.Item value="library" onClick={openLibrary}>
        {t('widgets.workflow.libraryWithEllipsis')}
      </Menu.Item>
      <Menu.Item value="save-to-library" onClick={saveToLibrary}>
        {t('widgets.workflow.saveToLibraryWithEllipsis')}
      </Menu.Item>
      {canUpdateSource ? (
        <Menu.Item value="update-template" onClick={updateTemplate}>
          {t('widgets.workflow.updateTemplateWithEllipsis')}
        </Menu.Item>
      ) : null}
      <Menu.Item value="import" onClick={requestWorkflowImport}>
        {t('widgets.workflow.importJsonWithEllipsis')}
      </Menu.Item>
      <Menu.Item value="export" onClick={exportWorkflow}>
        {t('widgets.workflow.exportJson')}
      </Menu.Item>
      <Menu.Item value="copy" onClick={copyWorkflow}>
        {t('widgets.workflow.copyJson')}
      </Menu.Item>
      <Menu.Item value="new" onClick={newWorkflow}>
        {t('widgets.workflow.newWorkflow')}
      </Menu.Item>
    </Menu.ItemGroup>
  );
};

const selectPersistenceView = (persistence: WorkflowProjectPersistence) => ({
  hasLocalRecovery: persistence.hasLocalRecovery,
  status: persistence.status,
});

export const WorkflowHeaderActions = ({ region }: WorkflowWidgetViewProps) => {
  const { t } = useTranslation();
  const activeWorkflowId = useWorkflowProjectSelector((project) => project.activeWorkflowId);
  const { hasLocalRecovery, status } = useWorkflowPersistenceSelector(selectPersistenceView);
  const openAddNode = useCallback(() => setAddNodeOpen(true), []);
  const openWorkflowLibrary = useCallback(() => setWorkflowLibraryOpen(true), []);
  const saveToLibrary = useCallback(
    () => requestWorkflowPublication({ kind: 'save-as-new', workflowId: activeWorkflowId }),
    [activeWorkflowId]
  );
  const view = PERSISTENCE_VIEW[status];
  // A pending save without browser recovery has no safety net at all; say so rather than showing the same cloud.
  const persistenceLabel =
    status === 'pending' && !hasLocalRecovery ? t('widgets.workflow.persistencePendingNoRecovery') : t(view.labelKey);

  return (
    <HStack gap="0.5">
      {region === 'center' ? (
        <Tooltip content={t('widgets.workflow.addNode')}>
          <IconButton
            aria-label={t('widgets.workflow.addNode')}
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={openAddNode}
          >
            <Icon as={PlusIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      ) : null}
      {/* Center labels already open the project's workflows; other regions need this separate trigger and leave
          publication and persistence to the editor header. */}
      {region === 'center' ? null : (
        <Tooltip content={t('widgets.workflow.library')}>
          <IconButton
            aria-label={t('widgets.workflow.library')}
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={openWorkflowLibrary}
          >
            <Icon as={LibraryIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      )}
      {region === 'center' ? (
        <>
          <Tooltip content={t('widgets.workflow.saveToLibraryWithEllipsis')}>
            <IconButton
              aria-label={t('widgets.workflow.saveToLibraryWithEllipsis')}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={saveToLibrary}
            >
              <Icon as={BookmarkIcon} boxSize="3.5" />
            </IconButton>
          </Tooltip>
          <Tooltip content={persistenceLabel}>
            <Box
              alignItems="center"
              aria-label={persistenceLabel}
              boxSize="6"
              color={view.color}
              data-persistence-status={status}
              display="flex"
              justifyContent="center"
              role="status"
            >
              <Icon as={view.icon} boxSize="3.5" />
            </Box>
          </Tooltip>
        </>
      ) : null}
    </HStack>
  );
};

export const WorkflowDialogHost = () => {
  const { addWorkflow, editGraph, renameWorkflow } = useProjectGraphCommands();
  const { project: projectStore } = useWorkflowUi();
  const notify = useWorkflowNotifications();
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const addNodeConnection = workflowUiStore.useSelector((snapshot) => snapshot.addNodeConnection);
  const addNodePosition = workflowUiStore.useSelector((snapshot) => snapshot.addNodePosition);
  const importRequestCount = workflowUiStore.useSelector((snapshot) => snapshot.importRequestCount);
  const isAddNodeOpen = workflowUiStore.useSelector((snapshot) => snapshot.isAddNodeOpen);
  const isLibraryOpen = workflowUiStore.useSelector((snapshot) => snapshot.isLibraryOpen);
  const lastImportRequestRef = useRef(importRequestCount);
  // A rename request opens the dialog once, for the workflow it named, and only while that workflow is the active
  // one; closing records the request as handled.
  const renameRequest = workflowUiStore.useSelector((snapshot) => snapshot.renameRequest);
  const [handledRenameRequestId, setHandledRenameRequestId] = useState(0);
  const activeWorkflowId = useWorkflowProjectSelector((project) => project.activeWorkflowId);
  const renameTargetId = renameRequest?.workflowId ?? null;
  const renameTargetName = useWorkflowProjectSelector(
    (project) => project.workflows.find((entry) => entry.document.id === renameTargetId)?.document.name ?? ''
  );
  const isRenameOpen =
    renameRequest !== null &&
    renameRequest.requestId > handledRenameRequestId &&
    renameRequest.workflowId === activeWorkflowId;

  // A request whose workflow stopped being active is over; it must not come back when that workflow returns.
  if (renameRequest !== null && renameRequest.requestId > handledRenameRequestId && !isRenameOpen) {
    setHandledRenameRequestId(renameRequest.requestId);
  }
  const closeRename = useCallback(
    () => setHandledRenameRequestId(renameRequest?.requestId ?? 0),
    [renameRequest?.requestId]
  );
  const submitRename = useCallback(
    (name: string) => {
      if (renameTargetId !== null) {
        renameWorkflow(renameTargetId, name);
      }
    },
    [renameTargetId, renameWorkflow]
  );

  // Editor session state (viewports) is keyed by workflow; release it once its workflow leaves the project.
  useMountEffect(() => {
    let lastMembership = '';

    return projectStore.subscribe(() => {
      const snapshot = projectStore.getSnapshot();
      const liveWorkflowIds = snapshot.workflows.map((entry) => entry.document.id);
      const membership = `${snapshot.id}\0${liveWorkflowIds.join('\0')}`;

      if (membership === lastMembership) {
        return;
      }

      lastMembership = membership;
      releaseWorkflowViewportsExcept(snapshot.id, liveWorkflowIds);
    });
  });

  useEffect(() => {
    if (importRequestCount > lastImportRequestRef.current) {
      fileInputRef.current?.click();
    }

    lastImportRequestRef.current = importRequestCount;
  }, [importRequestCount]);

  const getInsertPosition = useCallback((): XYPosition => {
    if (addNodePosition) {
      return addNodePosition;
    }

    const instance = getWorkflowFlowInstance();
    const center = instance
      ? instance.screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 })
      : { x: 0, y: 0 };

    // Slight scatter so repeated inserts do not stack perfectly.
    return { x: center.x + (Math.random() - 0.5) * 80, y: center.y + (Math.random() - 0.5) * 80 };
  }, [addNodePosition]);

  const addNode = useCallback(
    (template: InvocationTemplate) => {
      const node = buildInvocationNode(template, getInsertPosition());

      if (!addNodeConnection) {
        editGraph({ node, type: 'addNode' });
        return;
      }

      if (addNodeConnection.kind === 'source') {
        const targetInput =
          addNodeConnection.sourceHandle === LOOP_LINKAGE_FIELD && template.type === 'for_return'
            ? template.inputs[LOOP_LINKAGE_FIELD]
            : template.type === 'for_return'
              ? template.inputs.output
              : getCompatibleInputTemplate(template, addNodeConnection.sourceType);

        if (!targetInput) {
          editGraph({ node, type: 'addNode' });
          return;
        }

        const edge = {
          id: createWorkflowId('edge'),
          source: addNodeConnection.sourceNodeId,
          sourceHandle: addNodeConnection.sourceHandle,
          target: node.id,
          targetHandle: targetInput.name,
          type:
            addNodeConnection.sourceHandle === LOOP_LINKAGE_FIELD && template.type === 'for_return'
              ? ('loop_linkage' as const)
              : ('default' as const),
        };

        const currentGraph = projectStore.getSnapshot().projectGraph;
        const sourceNode = currentGraph.nodes.find((candidate) => candidate.id === addNodeConnection.sourceNodeId);
        const resolvedSource =
          sourceNode?.type === 'connector'
            ? resolveConnectorSource(sourceNode.id, currentGraph.nodes, currentGraph.edges)
            : sourceNode?.type === 'invocation'
              ? { fieldName: addNodeConnection.sourceHandle, nodeId: sourceNode.id, type: null }
              : null;
        const resolvedSourceNode = currentGraph.nodes.find((candidate) => candidate.id === resolvedSource?.nodeId);
        const shouldAddLoopLinkage = shouldAddForReturnLoopLinkage(
          template.type,
          resolvedSource,
          resolvedSourceNode,
          currentGraph.edges
        );

        const edges = [edge];
        if (shouldAddLoopLinkage && resolvedSourceNode?.type === 'invocation') {
          edges.push({
            id: createWorkflowId('edge'),
            source: resolvedSourceNode.id,
            sourceHandle: LOOP_LINKAGE_FIELD,
            target: node.id,
            targetHandle: LOOP_LINKAGE_FIELD,
            type: 'loop_linkage',
          });
        }

        editGraph({ edge: edges, node, type: 'addNodeAndEdge' });
        return;
      }

      const sourceOutput =
        addNodeConnection.targetHandle === LOOP_LINKAGE_FIELD && template.type === 'for'
          ? template.outputs[LOOP_LINKAGE_FIELD]
          : getCompatibleOutputTemplate(template, addNodeConnection.targetType);

      if (!sourceOutput) {
        editGraph({ node, type: 'addNode' });
        return;
      }

      editGraph({
        edge: {
          id: createWorkflowId('edge'),
          source: node.id,
          sourceHandle: sourceOutput.name,
          target: addNodeConnection.targetNodeId,
          targetHandle: addNodeConnection.targetHandle,
          type:
            addNodeConnection.targetHandle === LOOP_LINKAGE_FIELD && template.type === 'for'
              ? 'loop_linkage'
              : 'default',
        },
        node,
        type: 'addNodeAndEdge',
      });
    },
    [addNodeConnection, editGraph, getInsertPosition, projectStore]
  );

  const addNote = useCallback(() => {
    editGraph({ node: buildNotesNode(getInsertPosition()), type: 'addNode' });
  }, [editGraph, getInsertPosition]);

  const addConnector = useCallback(() => {
    const node = buildConnectorNode(getInsertPosition());

    if (!addNodeConnection) {
      editGraph({ node, type: 'addNode' });
      return;
    }

    editGraph({
      edge:
        addNodeConnection.kind === 'source'
          ? {
              id: createWorkflowId('edge'),
              source: addNodeConnection.sourceNodeId,
              sourceHandle: addNodeConnection.sourceHandle,
              target: node.id,
              targetHandle: CONNECTOR_INPUT_HANDLE,
              type: 'default',
            }
          : {
              id: createWorkflowId('edge'),
              source: node.id,
              sourceHandle: CONNECTOR_OUTPUT_HANDLE,
              target: addNodeConnection.targetNodeId,
              targetHandle: addNodeConnection.targetHandle,
              type: 'default',
            },
      node,
      type: 'addNodeAndEdge',
    });
  }, [addNodeConnection, editGraph, getInsertPosition]);

  const addCurrentImage = useCallback(() => {
    editGraph({ node: buildCurrentImageNode(getInsertPosition()), type: 'addNode' });
  }, [editGraph, getInsertPosition]);

  // A file is portable: whatever id it carries names nothing this project may write to.
  const importFile = useCallback(
    (file: File) => {
      file
        .text()
        .then((text) => {
          const { document: parsed, warnings: parseWarnings } = parseWorkflowJson(JSON.parse(text));
          const { document, warnings: updateWarnings } = updateLoadedWorkflowNodes(parsed, t);
          const warnings = [...parseWarnings, ...updateWarnings];

          addWorkflow(document, {
            label: t('widgets.workflow.importedLabel', { name: file.name }),
            reusePlaceholder: true,
          });

          for (const warning of warnings) {
            notify.info(t('widgets.workflow.importWarning'), warning);
          }
        })
        .catch((error: unknown) => {
          notify.error(
            t('widgets.workflow.importFailed'),
            error instanceof Error ? error.message : t('widgets.workflow.importInvalid')
          );
        });
    },
    [addWorkflow, notify, t]
  );
  const handleImportFile = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];

      event.currentTarget.value = '';

      if (file) {
        importFile(file);
      }
    },
    [importFile]
  );

  return (
    <>
      <CallSavedWorkflowSyncRuntime />
      <input ref={fileInputRef} accept=".json,application/json" hidden type="file" onChange={handleImportFile} />
      <AddNodeDialog
        connectionFilter={addNodeConnection}
        isOpen={isAddNodeOpen}
        onAddCurrentImage={addCurrentImage}
        onAddConnector={addConnector}
        onAddNode={addNode}
        onAddNote={addNote}
        onOpenChange={setAddNodeOpen}
      />
      <WorkflowLibraryDialog isOpen={isLibraryOpen} onOpenChange={setWorkflowLibraryOpen} />
      <RenameDialog
        key={renameRequest?.requestId ?? 0}
        initialName={renameTargetName}
        isOpen={isRenameOpen}
        label={t('workflowLibrary.workflowName')}
        submitLabel={t('workflowLibrary.rename')}
        title={t('workflowLibrary.renameTitle')}
        onClose={closeRename}
        onSubmit={submitRename}
      />
      <WorkflowPublicationHost />
      <PendingWorkflowLoader />
    </>
  );
};
