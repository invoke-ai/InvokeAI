import type { InvocationTemplate, XYPosition } from '@features/workflow/contracts';

import { HStack, Icon, Input, Menu, Portal, Stack, Text } from '@chakra-ui/react';
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
} from '@features/workflow/utility';
import { IconButton, ConfirmDialog, Tooltip } from '@platform/ui';
import { HistoryIcon, LibraryIcon, PlusIcon } from 'lucide-react';
import { useCallback, useEffect, useId, useMemo, useRef, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowWidgetLabelProps, WorkflowWidgetViewProps } from './contracts';

import { AddNodeDialog } from './editor/AddNodeDialog';
import { getWorkflowFlowInstance } from './editor/flowInstanceStore';
import { WorkflowLibraryDialog } from './library/WorkflowLibraryDialog';
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
 * The workflow widget's frame chrome. The label renders the editable
 * `Workflow / [name]` title; quick actions (add node, library, history) are
 * header icon buttons; everything else contributes to the shared widget
 * actions menu via the manifest's `headerMenu`. Dialogs live here (always
 * mounted) and are driven through `workflowUiStore`.
 */

export const WorkflowWidgetLabel = ({ region }: WorkflowWidgetLabelProps) => {
  const { t } = useTranslation();
  const workflowName = useWorkflowProjectSelector((project) => project.projectGraph.name);
  const { editGraph } = useProjectGraphCommands();
  const changeWorkflowName = useCallback(
    (event: ChangeEvent<HTMLInputElement>) =>
      editGraph({ patch: { name: event.currentTarget.value }, type: 'setMetadata' }),
    [editGraph]
  );

  if (region !== 'center') {
    return (
      <Text fontSize="xs" fontWeight="700">
        {t('widgets.labels.workflow')}
      </Text>
    );
  }

  return (
    <HStack flex="1" gap="1" minW="0">
      <Text flexShrink={0} fontSize="xs" fontWeight="700">
        {t('widgets.labels.workflow')}
      </Text>
      <Text color="fg.subtle" flexShrink={0} fontSize="xs">
        /
      </Text>
      <Input
        aria-label={t('widgets.workflow.name')}
        fontSize="xs"
        fontWeight="600"
        h="6"
        maxW="16rem"
        placeholder={t('widgets.workflow.untitled')}
        size="2xs"
        value={workflowName}
        variant="flushed"
        onChange={changeWorkflowName}
      />
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
  const { restoreSnapshot } = useProjectGraphCommands();
  const restorableHistory = graphHistory.filter((entry) => entry.document);
  const historyTriggerId = useId();
  const openAddNode = useCallback(() => setAddNodeOpen(true), []);
  const openWorkflowLibrary = useCallback(() => setWorkflowLibraryOpen(true), []);
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
                      <Menu.ItemText fontSize="xs" truncate>
                        {entry.label}
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
  const notify = useWorkflowNotifications();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const addNodeConnection = workflowUiStore.useSelector((snapshot) => snapshot.addNodeConnection);
  const addNodePosition = workflowUiStore.useSelector((snapshot) => snapshot.addNodePosition);
  const importRequestCount = workflowUiStore.useSelector((snapshot) => snapshot.importRequestCount);
  const isAddNodeOpen = workflowUiStore.useSelector((snapshot) => snapshot.isAddNodeOpen);
  const isLibraryOpen = workflowUiStore.useSelector((snapshot) => snapshot.isLibraryOpen);
  const isNewWorkflowConfirmOpen = workflowUiStore.useSelector((snapshot) => snapshot.isNewWorkflowConfirmOpen);
  const lastImportRequestRef = useRef(importRequestCount);

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
