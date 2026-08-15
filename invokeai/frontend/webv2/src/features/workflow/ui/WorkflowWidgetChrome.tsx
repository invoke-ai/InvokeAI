import type { InvocationTemplate, XYPosition } from '@features/workflow/contracts';

import { Box, HStack, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { updateLibraryWorkflow } from '@features/workflow/queries';
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
import {
  CloudAlertIcon,
  CloudCheckIcon,
  CloudUploadIcon,
  HistoryIcon,
  LibraryIcon,
  PlusIcon,
  RefreshCwIcon,
} from 'lucide-react';
import { useCallback, useEffect, useId, useMemo, useRef, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowWidgetLabelProps, WorkflowWidgetViewProps } from './contracts';

import { AddNodeDialog } from './editor/AddNodeDialog';
import { getWorkflowFlowInstance } from './editor/flowInstanceStore';
import { createLibraryAutosaver, type LibrarySyncStatus } from './library/libraryAutosave';
import { registerLibraryGraphSyncedHandler, releaseLibraryGraphSyncedHandler } from './library/librarySyncBridge';
import { useSaveWorkflowToLibrary } from './library/useSaveWorkflowToLibrary';
import { WorkflowLibraryDialog } from './library/WorkflowLibraryDialog';
import { setWorkflowLibrarySyncStatus, workflowLibrarySyncStore } from './library/workflowLibrarySyncStore';
import { PendingLibraryWorkflowLoader } from './PendingLibraryWorkflowLoader';
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
 * The workflow widget's frame chrome. The label renders the `Workflow / [name]`
 * title, where the name is the library trigger; quick actions (add node,
 * library, history) are header icon buttons; everything else contributes to the
 * shared widget actions menu via the manifest's `headerMenu`. Dialogs live here
 * (always mounted) and are driven through `workflowUiStore`.
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

  // In the center the region's view selector already names the widget, so the
  // label slot contributes only the `/ [name]` continuation. The name opens the
  // library rather than editing in place: switching workflows is the thing
  // reached for from a header, and renaming already lives in the details tab —
  // an inline field here only invited stray keystrokes into the graph's name.
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
          <Icon as={LibraryIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
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
  const { saveSnapshot } = useProjectGraphCommands();
  const notify = useWorkflowNotifications();

  const openDetailsPanel = useCallback(() => {
    widgets.open({ region: 'left', widgetId: 'workflow' });
    widgets.patchValues('workflow', { editTab: 'details', panelMode: 'edit' });
  }, [widgets]);
  const exportWorkflow = useCallback(() => downloadWorkflowJson(getProjectGraph()), [getProjectGraph]);
  const copyWorkflow = useCallback(() => {
    copyWorkflowJson(getProjectGraph())
      .then(() => notify.success(t('widgets.workflow.copyJsonSuccess')))
      .catch(() => notify.error(t('widgets.workflow.copyJsonFailed')));
  }, [getProjectGraph, notify, t]);
  const createNewWorkflow = useCallback(() => setNewWorkflowConfirmOpen(true), []);

  return (
    <Menu.ItemGroup>
      <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
        {t('widgets.labels.workflow')}
      </Menu.ItemGroupLabel>
      <Menu.Item value="snapshot" onClick={saveSnapshot}>
        {t('widgets.workflow.saveGraphSnapshot')}
      </Menu.Item>
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
      <Menu.Item color="fg.error" value="new" onClick={createNewWorkflow}>
        {t('widgets.workflow.newWorkflowWithEllipsis')}
      </Menu.Item>
    </Menu.ItemGroup>
  );
};

export const WorkflowHeaderActions = ({ region }: WorkflowWidgetViewProps) => {
  const { t } = useTranslation();
  const graphHistory = useWorkflowProjectSelector((project) => project.graphHistory);
  const libraryWorkflowId = useWorkflowProjectSelector((project) => project.projectGraph.libraryWorkflowId);
  const syncStatus = workflowLibrarySyncStore.useSelector((snapshot) => snapshot.status);
  const { restoreSnapshot } = useProjectGraphCommands();
  const { saveToLibrary } = useSaveWorkflowToLibrary();
  const restorableHistory = graphHistory.filter((entry) => entry.document);
  const historyTriggerId = useId();
  const openAddNode = useCallback(() => setAddNodeOpen(true), []);
  const openWorkflowLibrary = useCallback(() => setWorkflowLibraryOpen(true), []);
  const handleSaveToLibrary = useCallback(() => void saveToLibrary(), [saveToLibrary]);
  const historyIds = useMemo(() => ({ trigger: historyTriggerId }), [historyTriggerId]);
  const historyPositioning = useMemo(() => ({ placement: 'bottom-end' as const }), []);
  const restoreHistoryEntry = useCallback(
    (details: { value: string }) => restoreSnapshot(details.value),
    [restoreSnapshot]
  );

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
      {/* Center already reaches the library through the header label, which
          names the current workflow; a second identical trigger beside it would
          be pure duplication. Other regions render a plain `Workflow` label, so
          they still need this. */}
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
      <Menu.Root ids={historyIds} positioning={historyPositioning} onSelect={restoreHistoryEntry}>
        <Tooltip content={t('widgets.workflow.graphHistorySnapshots')} ids={historyIds}>
          <Menu.Trigger asChild>
            <IconButton
              aria-label={t('widgets.workflow.graphHistorySnapshots')}
              color="fg.muted"
              disabled={restorableHistory.length === 0}
              size="2xs"
              variant="ghost"
            >
              <Icon as={HistoryIcon} boxSize="3.5" />
            </IconButton>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <Menu.Content maxH="18rem" minW="16rem" overflowY="auto">
              <Menu.ItemGroup>
                <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                  {t('widgets.workflow.graphHistorySnapshots')}
                </Menu.ItemGroupLabel>
                {restorableHistory.map((entry) => (
                  <Menu.Item key={entry.id} value={entry.id}>
                    <Stack gap="0" minW="0">
                      <Menu.ItemText fontSize="xs">
                        <MiddleTruncate as="span" text={entry.label} />
                      </Menu.ItemText>
                      <Text color="fg.subtle" fontSize="2xs">
                        {new Date(entry.createdAt).toLocaleString()}
                      </Text>
                    </Stack>
                  </Menu.Item>
                ))}
              </Menu.ItemGroup>
            </Menu.Content>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </HStack>
  );
};

export const WorkflowDialogHost = () => {
  const { editGraph, replace } = useProjectGraphCommands();
  const { project: projectStore } = useWorkflowUi();
  const notify = useWorkflowNotifications();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const addNodeConnection = workflowUiStore.useSelector((snapshot) => snapshot.addNodeConnection);
  const addNodePosition = workflowUiStore.useSelector((snapshot) => snapshot.addNodePosition);
  const importRequestCount = workflowUiStore.useSelector((snapshot) => snapshot.importRequestCount);
  const isAddNodeOpen = workflowUiStore.useSelector((snapshot) => snapshot.isAddNodeOpen);
  const isLibraryOpen = workflowUiStore.useSelector((snapshot) => snapshot.isLibraryOpen);
  const isNewWorkflowConfirmOpen = workflowUiStore.useSelector((snapshot) => snapshot.isNewWorkflowConfirmOpen);
  const lastImportRequestRef = useRef(importRequestCount);

  // Library autosave: bound project graphs save themselves back to the
  // library after edits settle. This host is always mounted alongside the
  // workflow widget, so one autosaver instance tracks the graph for the
  // widget's whole lifetime — but that instance is created AND disposed
  // within a single mount effect (held in a ref, not a `useState`
  // initializer). A `useState` initializer only runs once per fiber, while
  // its disposal lived in a separate effect's cleanup; React StrictMode's
  // dev-only mount→cleanup→mount simulation ran that cleanup once without
  // ever re-running the initializer, permanently disposing the one instance
  // still in use (autosave silently no-oped for the rest of the session).
  // Creating and disposing in the same effect makes the simulation produce a
  // fresh, live instance instead. `read()` pulls straight from the project
  // store's synchronous snapshot (`projectStore.getSnapshot()`, the same
  // accessor `useWorkflowProjectSelector` subscribes through) rather than a
  // component-owned ref, so it is always current whenever the autosaver's
  // debounce timer or `flush()` calls it. The same effect subscribes
  // directly to the project store to notify the autosaver of edits (skipping
  // the initial snapshot so loading a workflow does not itself count as an
  // edit) — a plain subscription rather than a selector, since nothing here
  // needs to re-render on graph edits, only to poke the autosaver.
  //
  // `hostScope` guards `onStatus`: account rotation aborts the in-flight
  // save's signal and synchronously resets the (account-owned)
  // `workflowLibrarySyncStore` back to 'idle', but the aborted write's
  // rejection lands a tick later, after that reset. Without this guard,
  // `runSave()`'s `.catch` would still write 'error' into the *next*
  // account's store — `save()`'s own `assertAccountScopeCurrent` throw stops
  // the write from being trusted, but does not by itself stop that throw's
  // `onStatus('error')` from landing. `hostScope` is captured once per mount
  // — safe because this host remounts every account epoch (keyed by
  // `session.accountEpoch` above it in the tree), so a mount-captured scope
  // never outlives the account it was captured for.
  useEffect(() => {
    const hostScope = captureAccountScope();
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
        // Same scope discipline as the manual save paths: never let a
        // debounced write land in the next account's library.
        const owner = captureAccountScope();
        await updateLibraryWorkflow(workflowId, serialized, owner.signal);
        assertAccountScopeCurrent(owner);
      },
    });

    let lastGraph = projectStore.getSnapshot().projectGraph;
    const unsubscribe = projectStore.subscribe(() => {
      const graph = projectStore.getSnapshot().projectGraph;
      if (graph !== lastGraph) {
        lastGraph = graph;
        autosaver.notifyGraphChanged();
      }
    });

    const handler = (serialized: Record<string, unknown>) => autosaver.markSynced(serialized);

    registerLibraryGraphSyncedHandler(handler);

    return () => {
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
        const targetInput = getCompatibleInputTemplate(template, addNodeConnection.sourceType);

        if (!targetInput) {
          editGraph({ node, type: 'addNode' });
          return;
        }

        editGraph({
          edge: {
            id: createWorkflowId('edge'),
            source: addNodeConnection.sourceNodeId,
            sourceHandle: addNodeConnection.sourceHandle,
            target: node.id,
            targetHandle: targetInput.name,
            type: 'default',
          },
          node,
          type: 'addNodeAndEdge',
        });
        return;
      }

      const sourceOutput = getCompatibleOutputTemplate(template, addNodeConnection.targetType);

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
          type: 'default',
        },
        node,
        type: 'addNodeAndEdge',
      });
    },
    [addNodeConnection, editGraph, getInsertPosition]
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
          const { document, warnings } = parseWorkflowJson(JSON.parse(text));

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
    [notify, replace]
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
      <PendingLibraryWorkflowLoader />
      <ConfirmDialog
        body="Replace the project graph with an empty workflow? The current graph is saved to graph history first."
        confirmLabel="New workflow"
        isOpen={isNewWorkflowConfirmOpen}
        title="New workflow"
        onClose={closeNewWorkflowConfirm}
        onConfirm={confirmNewWorkflow}
      />
    </>
  );
};
