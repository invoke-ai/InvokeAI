import type { WidgetLabelProps, WidgetViewProps } from '@workbench/types';
import type { InvocationTemplate, XYPosition } from '@workbench/workflows/types';

import { HStack, Icon, Input, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { IconButton, ConfirmDialog, Tooltip } from '@workbench/components/ui';
import { useNotify } from '@workbench/useNotify';
import { useActiveProjectSelector, useWorkbenchDispatch, useWorkbenchStore } from '@workbench/WorkbenchContext';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from '@workbench/workflows/connectors';
import {
  buildConnectorNode,
  buildCurrentImageNode,
  buildInvocationNode,
  buildNotesNode,
  createProjectGraph,
  createWorkflowId,
} from '@workbench/workflows/document';
import { getCompatibleInputTemplate, getCompatibleOutputTemplate } from '@workbench/workflows/validation';
import { parseWorkflowJson } from '@workbench/workflows/workflowJson';
import { HistoryIcon, LibraryIcon, PlusIcon } from 'lucide-react';
import { useEffect, useId, useRef, type ChangeEvent } from 'react';

import { AddNodeDialog } from './editor/AddNodeDialog';
import { getWorkflowFlowInstance } from './editor/flowInstanceStore';
import { WorkflowLibraryDialog } from './library/WorkflowLibraryDialog';
import { copyWorkflowJson, downloadWorkflowJson } from './workflowTransfer';
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

export const WorkflowWidgetLabel = ({ region }: WidgetLabelProps) => {
  const workflowName = useActiveProjectSelector((project) => project.projectGraph.name);
  const dispatch = useWorkbenchDispatch();

  if (region !== 'center') {
    return (
      <Text fontSize="xs" fontWeight="700">
        Workflow
      </Text>
    );
  }

  return (
    <HStack flex="1" gap="1" minW="0">
      <Text flexShrink={0} fontSize="xs" fontWeight="700">
        Workflow
      </Text>
      <Text color="fg.subtle" flexShrink={0} fontSize="xs">
        /
      </Text>
      <Input
        aria-label="Workflow name"
        fontSize="xs"
        fontWeight="600"
        h="6"
        maxW="16rem"
        placeholder="Untitled Workflow"
        size="2xs"
        value={workflowName}
        variant="flushed"
        onChange={(event: ChangeEvent<HTMLInputElement>) =>
          dispatch({
            action: { patch: { name: event.currentTarget.value }, type: 'setMetadata' },
            type: 'applyProjectGraphAction',
          })
        }
      />
    </HStack>
  );
};

/** Entries contributed to the shared widget actions menu. */
export const WorkflowMenuItems = (_props: WidgetViewProps) => {
  const store = useWorkbenchStore();
  const dispatch = useWorkbenchDispatch();
  const notify = useNotify();

  const openDetailsPanel = () => {
    dispatch({ region: 'left', type: 'openRegionWidget', widgetId: 'workflow' });
    dispatch({ type: 'patchWidgetValues', values: { editTab: 'details', panelMode: 'edit' }, widgetId: 'workflow' });
  };

  return (
    <Menu.ItemGroup>
      <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
        Workflow
      </Menu.ItemGroupLabel>
      <Menu.Item value="snapshot" onClick={() => dispatch({ type: 'saveProjectGraphSnapshot' })}>
        Save graph snapshot
      </Menu.Item>
      <Menu.Item value="details" onClick={openDetailsPanel}>
        Workflow details…
      </Menu.Item>
      <Menu.Item value="import" onClick={requestWorkflowImport}>
        Import workflow JSON…
      </Menu.Item>
      <Menu.Item value="export" onClick={() => downloadWorkflowJson(store.getSnapshot().activeProject.projectGraph)}>
        Export workflow JSON
      </Menu.Item>
      <Menu.Item
        value="copy"
        onClick={() => {
          copyWorkflowJson(store.getSnapshot().activeProject.projectGraph)
            .then(() => notify.success('Workflow JSON copied'))
            .catch(() => notify.error('Failed to copy workflow JSON'));
        }}
      >
        Copy workflow JSON
      </Menu.Item>
      <Menu.Item color="fg.error" value="new" onClick={() => setNewWorkflowConfirmOpen(true)}>
        New workflow…
      </Menu.Item>
    </Menu.ItemGroup>
  );
};

export const WorkflowHeaderActions = ({ region }: WidgetViewProps) => {
  const graphHistory = useActiveProjectSelector((project) => project.graphHistory);
  const dispatch = useWorkbenchDispatch();
  const restorableHistory = graphHistory.filter((entry) => entry.document);
  const historyTriggerId = useId();

  return (
    <HStack gap="0.5">
      {region === 'center' ? (
        <Tooltip content="Add node">
          <IconButton
            aria-label="Add node"
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={() => setAddNodeOpen(true)}
          >
            <Icon as={PlusIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      ) : null}
      <Tooltip content="Workflow library">
        <IconButton
          aria-label="Workflow library"
          color="fg.muted"
          size="2xs"
          variant="ghost"
          onClick={() => setWorkflowLibraryOpen(true)}
        >
          <Icon as={LibraryIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
      <Menu.Root ids={{ trigger: historyTriggerId }} positioning={{ placement: 'bottom-end' }}>
        <Tooltip content="Graph history snapshots" ids={{ trigger: historyTriggerId }}>
          <Menu.Trigger asChild>
            <IconButton
              aria-label="Graph history snapshots"
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
                  Graph History Snapshots
                </Menu.ItemGroupLabel>
                {restorableHistory.map((entry) => (
                  <Menu.Item
                    key={entry.id}
                    value={entry.id}
                    onClick={() => dispatch({ snapshotId: entry.id, type: 'restoreProjectGraphSnapshot' })}
                  >
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
  const dispatch = useWorkbenchDispatch();
  const notify = useNotify();
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

  const getInsertPosition = (): XYPosition => {
    if (addNodePosition) {
      return addNodePosition;
    }

    const instance = getWorkflowFlowInstance();
    const center = instance
      ? instance.screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 })
      : { x: 0, y: 0 };

    // Slight scatter so repeated inserts do not stack perfectly.
    return { x: center.x + (Math.random() - 0.5) * 80, y: center.y + (Math.random() - 0.5) * 80 };
  };

  const addNode = (template: InvocationTemplate) => {
    const node = buildInvocationNode(template, getInsertPosition());

    if (!addNodeConnection) {
      dispatch({
        action: { node, type: 'addNode' },
        type: 'applyProjectGraphAction',
      });
      return;
    }

    if (addNodeConnection.kind === 'source') {
      const targetInput = getCompatibleInputTemplate(template, addNodeConnection.sourceType);

      if (!targetInput) {
        dispatch({
          action: { node, type: 'addNode' },
          type: 'applyProjectGraphAction',
        });
        return;
      }

      dispatch({
        action: {
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
        },
        type: 'applyProjectGraphAction',
      });
      return;
    }

    const sourceOutput = getCompatibleOutputTemplate(template, addNodeConnection.targetType);

    if (!sourceOutput) {
      dispatch({
        action: { node, type: 'addNode' },
        type: 'applyProjectGraphAction',
      });
      return;
    }

    dispatch({
      action: {
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
      },
      type: 'applyProjectGraphAction',
    });
  };

  const addNote = () => {
    dispatch({
      action: { node: buildNotesNode(getInsertPosition()), type: 'addNode' },
      type: 'applyProjectGraphAction',
    });
  };

  const addConnector = () => {
    const node = buildConnectorNode(getInsertPosition());

    if (!addNodeConnection) {
      dispatch({
        action: { node, type: 'addNode' },
        type: 'applyProjectGraphAction',
      });
      return;
    }

    dispatch({
      action: {
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
      },
      type: 'applyProjectGraphAction',
    });
  };

  const addCurrentImage = () => {
    dispatch({
      action: { node: buildCurrentImageNode(getInsertPosition()), type: 'addNode' },
      type: 'applyProjectGraphAction',
    });
  };

  const importFile = (file: File) => {
    file
      .text()
      .then((text) => {
        const { document, warnings } = parseWorkflowJson(JSON.parse(text));

        dispatch({ document, label: `Imported "${file.name}"`, type: 'replaceProjectGraph' });

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
  };

  return (
    <>
      <input
        ref={fileInputRef}
        accept=".json,application/json"
        hidden
        type="file"
        onChange={(event: ChangeEvent<HTMLInputElement>) => {
          const file = event.currentTarget.files?.[0];

          event.currentTarget.value = '';

          if (file) {
            importFile(file);
          }
        }}
      />
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
        onClose={() => setNewWorkflowConfirmOpen(false)}
        onConfirm={() => {
          dispatch({
            document: createProjectGraph(createWorkflowId('project-graph')),
            label: 'New workflow',
            type: 'replaceProjectGraph',
          });
        }}
      />
    </>
  );
};
