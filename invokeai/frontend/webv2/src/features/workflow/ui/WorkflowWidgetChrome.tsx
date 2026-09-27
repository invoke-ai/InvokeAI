import type { InvocationTemplate, XYPosition } from '@features/workflow/contracts';

import { Box, HStack, Icon, Menu, Text } from '@chakra-ui/react';
import { updateLoadedWorkflowNodes } from '@features/workflow/data/templates';
import { invalidateWorkflowLibraryCache, updateLibraryWorkflow } from '@features/workflow/queries';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  buildConnectorNode,
  buildCurrentImageNode,
  buildInvocationNode,
  buildNotesNode,
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  createProjectGraph,
  createWorkflowId,
  getCompatibleInputTemplate,
  getCompatibleOutputTemplate,
  hasMultipleWorkflowReturnNodes,
  LOOP_LINKAGE_FIELD,
  resolveConnectorSource,
  shouldAddForReturnLoopLinkage,
  parseWorkflowJson,
  serializeWorkflowJson,
} from '@features/workflow/utility';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button, IconButton, ConfirmDialog, Tooltip } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { CloudAlertIcon, CloudCheckIcon, CloudUploadIcon, LibraryIcon, PlusIcon, RefreshCwIcon } from 'lucide-react';
import { useCallback, useEffect, useRef, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowWidgetLabelProps, WorkflowWidgetViewProps } from './contracts';

import { AddNodeDialog } from './editor/AddNodeDialog';
import { CallSavedWorkflowSyncRuntime } from './editor/CallSavedWorkflowSyncRuntime';
import { getWorkflowFlowInstance } from './editor/flowInstanceStore';
import { createLibraryAutosaver, type LibrarySyncStatus } from './library/libraryAutosave';
import { registerLibraryGraphSyncedHandler, releaseLibraryGraphSyncedHandler } from './library/librarySyncBridge';
import { useSaveWorkflowToLibrary } from './library/useSaveWorkflowToLibrary';
import { WorkflowLibraryDialog } from './library/WorkflowLibraryDialog';
import { setWorkflowLibrarySyncStatus, workflowLibrarySyncStore } from './library/workflowLibrarySyncStore';
import { PendingWorkflowLoader } from './PendingLibraryWorkflowLoader';
import { copyWorkflowJson, downloadWorkflowJson } from './workflowTransfer';
import {
  useWorkflowHostCommands,
  useWorkflowNotifications,
  useWorkflowProjectSelector,
  useWorkflowUi,
} from './WorkflowUiContext';
import {
  requestWorkflowImport,
  setAddNodeOpen,
  setNewWorkflowConfirmOpen,
  setWorkflowLibraryOpen,
  workflowUiStore,
} from './workflowUiStore';

/**
 * Keep dialogs mounted here and drive them through workflowUiStore; manifest slots contribute label, quick
 * actions, and shared menu actions.
 */

/** Icon + tooltip for each library sync status, shared by the sync control's error and non-error presentations. */
const SYNC_STATUS_VIEW: Record<LibrarySyncStatus, { icon: typeof CloudCheckIcon; tooltipKey: string }> = {
  dirty: { icon: RefreshCwIcon, tooltipKey: 'widgets.workflow.librarySyncSaving' },
  error: { icon: CloudAlertIcon, tooltipKey: 'widgets.workflow.librarySyncError' },
  idle: { icon: CloudCheckIcon, tooltipKey: 'widgets.workflow.librarySyncSaved' },
  saved: { icon: CloudCheckIcon, tooltipKey: 'widgets.workflow.librarySyncSaved' },
  saving: { icon: RefreshCwIcon, tooltipKey: 'widgets.workflow.librarySyncSaving' },
};

export const WorkflowWidgetLabel = ({ region }: WorkflowWidgetLabelProps) => {
  const { t } = useTranslation();
  const workflowName = useWorkflowProjectSelector((project) => project.projectGraph.name);
  const openWorkflowLibrary = useCallback(() => setWorkflowLibraryOpen(true), []);

  if (region !== 'center') {
    return (
      <Text fontSize="xs" fontWeight="700">
        {t('widgets.labels.workflow')}
      </Text>
    );
  }

  // Center chrome already names the widget; append only the library-opening workflow name and leave renaming in
  // Details.
  const displayName = workflowName || t('widgets.workflow.untitled');

  return (
    <HStack flex="1" gap="1" minW="0">
      <Text color="fg.subtle" flexShrink={0} fontSize="xs">
        /
      </Text>
      <Tooltip content={t('widgets.workflow.library')}>
        <Button
          aria-label={t('widgets.workflow.openLibrary', { name: displayName })}
          maxW="16rem"
          minW="0"
          size="2xs"
          variant="ghost"
          onClick={openWorkflowLibrary}
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
  const notify = useWorkflowNotifications();

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
  const createNewWorkflow = useCallback(() => setNewWorkflowConfirmOpen(true), []);

  return (
    <Menu.ItemGroup>
      <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
        {t('widgets.labels.workflow')}
      </Menu.ItemGroupLabel>
      <Menu.Item value="details" onClick={openDetailsPanel}>
        {t('widgets.workflow.detailsWithEllipsis')}
      </Menu.Item>
      <Menu.Item value="import" onClick={requestWorkflowImport}>
        {t('widgets.workflow.importJsonWithEllipsis')}
      </Menu.Item>
      <Menu.Item value="export" onClick={exportWorkflow}>
        {t('widgets.workflow.exportJson')}
      </Menu.Item>
      <Menu.Item value="copy" onClick={copyWorkflow}>
        {t('widgets.workflow.copyJson')}
      </Menu.Item>
      <Menu.Item data-danger="" value="new" onClick={createNewWorkflow}>
        {t('widgets.workflow.newWorkflowWithEllipsis')}
      </Menu.Item>
    </Menu.ItemGroup>
  );
};

export const WorkflowHeaderActions = ({ region }: WorkflowWidgetViewProps) => {
  const { t } = useTranslation();
  const libraryWorkflowId = useWorkflowProjectSelector((project) => project.projectGraph.libraryWorkflowId);
  const syncStatus = workflowLibrarySyncStore.useSelector((snapshot) => snapshot.status);
  const { saveToLibrary } = useSaveWorkflowToLibrary();
  const openAddNode = useCallback(() => setAddNodeOpen(true), []);
  const openWorkflowLibrary = useCallback(() => setWorkflowLibraryOpen(true), []);
  const handleSaveToLibrary = useCallback(() => void saveToLibrary(), [saveToLibrary]);

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
      {/* Center labels already open the library; other regions need this separate trigger. */}
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
      {libraryWorkflowId ? (
        <Tooltip content={t(SYNC_STATUS_VIEW[syncStatus].tooltipKey)}>
          {syncStatus === 'error' ? (
            <IconButton
              aria-label={t('widgets.workflow.librarySyncState')}
              color="fg.error"
              size="2xs"
              variant="ghost"
              onClick={handleSaveToLibrary}
            >
              <Icon as={SYNC_STATUS_VIEW.error.icon} boxSize="3.5" />
            </IconButton>
          ) : (
            <Box
              alignItems="center"
              aria-label={t(SYNC_STATUS_VIEW[syncStatus].tooltipKey)}
              boxSize="6"
              color="fg.subtle"
              display="flex"
              justifyContent="center"
              role="status"
            >
              <Icon as={SYNC_STATUS_VIEW[syncStatus].icon} boxSize="3.5" />
            </Box>
          )}
        </Tooltip>
      ) : (
        <Tooltip content={t('widgets.workflow.saveToLibrary')}>
          <IconButton
            aria-label={t('widgets.workflow.saveToLibrary')}
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={handleSaveToLibrary}
          >
            <Icon as={CloudUploadIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      )}
    </HStack>
  );
};

export const WorkflowDialogHost = () => {
  const { editGraph, replace } = useProjectGraphCommands();
  const { project: projectStore } = useWorkflowUi();
  const notify = useWorkflowNotifications();
  const { t } = useTranslation();
  const notifyRef = useRef(notify);
  const translationRef = useRef(t);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const addNodeConnection = workflowUiStore.useSelector((snapshot) => snapshot.addNodeConnection);
  const addNodePosition = workflowUiStore.useSelector((snapshot) => snapshot.addNodePosition);
  const importRequestCount = workflowUiStore.useSelector((snapshot) => snapshot.importRequestCount);
  const isAddNodeOpen = workflowUiStore.useSelector((snapshot) => snapshot.isAddNodeOpen);
  const isLibraryOpen = workflowUiStore.useSelector((snapshot) => snapshot.isLibraryOpen);
  const isNewWorkflowConfirmOpen = workflowUiStore.useSelector((snapshot) => snapshot.isNewWorkflowConfirmOpen);
  const lastImportRequestRef = useRef(importRequestCount);

  useEffect(() => {
    notifyRef.current = notify;
    translationRef.current = t;
  }, [notify, t]);

  // Create/dispose autosave in one mount lifecycle for StrictMode. Subscribe/read directly from the project store;
  // fence status callbacks by mount account scope to reject late-account errors.
  useEffect(() => {
    const hostScope = captureAccountScope();
    let autosaverActive = true;
    let duplicateReturnNotificationShown = false;
    const autosaver = createLibraryAutosaver({
      onStatus: (status) => {
        if (isAccountScopeCurrent(hostScope)) {
          setWorkflowLibrarySyncStatus(status);
        }
      },
      read: () => {
        const graph = projectStore.getSnapshot().projectGraph;
        return { libraryWorkflowId: graph.libraryWorkflowId, serialized: serializeWorkflowJson(graph) };
      },
      save: async (workflowId, serialized) => {
        assertAccountScopeCurrent(hostScope);
        const hasDuplicateWorkflowReturns = hasMultipleWorkflowReturnNodes(projectStore.getSnapshot().projectGraph);

        if (hasDuplicateWorkflowReturns) {
          if (autosaverActive && !duplicateReturnNotificationShown) {
            duplicateReturnNotificationShown = true;
            notifyRef.current.error(
              translationRef.current('workflowLibrary.saveFailed'),
              translationRef.current('workflowLibrary.multipleWorkflowReturnNodes')
            );
          }
          throw new Error('Workflow contains multiple workflow_return nodes.');
        }

        duplicateReturnNotificationShown = false;
        await updateLibraryWorkflow(workflowId, serialized, hostScope.signal);
        assertAccountScopeCurrent(hostScope);
        // The library dialog serves cached payloads and pages; a save changes both.
        invalidateWorkflowLibraryCache(workflowId);
      },
    });

    let lastGraph = projectStore.getSnapshot().projectGraph;
    const unsubscribe = projectStore.subscribe(() => {
      const graph = projectStore.getSnapshot().projectGraph;
      if (graph !== lastGraph) {
        lastGraph = graph;
        if (!hasMultipleWorkflowReturnNodes(graph)) {
          duplicateReturnNotificationShown = false;
        }
        autosaver.notifyGraphChanged();
      }
    });

    const handler = (serialized: Record<string, unknown>) => autosaver.markSynced(serialized);

    registerLibraryGraphSyncedHandler(handler);

    return () => {
      autosaverActive = false;
      unsubscribe();
      releaseLibraryGraphSyncedHandler(handler);
      autosaver.dispose();
    };
  }, [projectStore]);

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

  const importFile = useCallback(
    (file: File) => {
      file
        .text()
        .then((text) => {
          const { document: parsed, warnings: parseWarnings } = parseWorkflowJson(JSON.parse(text));
          const { document, warnings: updateWarnings } = updateLoadedWorkflowNodes(parsed, t);
          const warnings = [...parseWarnings, ...updateWarnings];

          replace(document, `Imported "${file.name}"`);

          for (const warning of warnings) {
            notify.info('Workflow import warning', warning);
          }
        })
        .catch((error: unknown) => {
          notify.error(
            'Failed to import workflow',
            error instanceof Error ? error.message : 'The file is not a valid workflow JSON.'
          );
        });
    },
    [notify, replace, t]
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
  const closeNewWorkflowConfirm = useCallback(() => setNewWorkflowConfirmOpen(false), []);
  const confirmNewWorkflow = useCallback(() => {
    replace(createProjectGraph(createWorkflowId('workflow')), 'New workflow');
  }, [replace]);

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
      <PendingWorkflowLoader />
      <ConfirmDialog
        body="Replace the project graph with an empty workflow? You can undo this change during this session. Save the current workflow to the library first if you need a permanent copy."
        confirmLabel="New workflow"
        isOpen={isNewWorkflowConfirmOpen}
        title="New workflow"
        onClose={closeNewWorkflowConfirm}
        onConfirm={confirmNewWorkflow}
      />
    </>
  );
};
