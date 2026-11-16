import type { ProjectGraphState } from '@features/workflow/core/types';
import type { WorkflowInvocationSourceId, WorkflowPreviewGraph } from '@features/workflow/ui/contracts';

import { Checkbox, chakra, Input, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { useAuthSession } from '@features/identity';
import { cloneProjectGraph, createWorkflowId } from '@features/workflow/core/document';
import { previewGraphToDocument } from '@features/workflow/core/graphToDocument';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import {
  discardUnresolvedWorkflowPublication,
  useWorkflowPublication,
  type WorkflowPublicationFailure,
} from '@features/workflow/ui/library/useWorkflowPublication';
import {
  useWorkflowGraphPreview,
  useWorkflowHostCommands,
  useWorkflowNotifications,
  useWorkflowUi,
} from '@features/workflow/ui/WorkflowUiContext';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import { downloadText } from '@platform/browser/downloadBlob';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, CloseButton, IconButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { Field } from '@platform/ui/Field';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { BookmarkIcon, CopyIcon, DownloadIcon, EllipsisIcon, GitForkIcon, PencilRulerIcon } from 'lucide-react';
import { useCallback, useRef, useState, type FormEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { getGraphJsonText } from './graphJson';

interface PreviewSaveRequest {
  document: ProjectGraphState;
  failure?: WorkflowPublicationFailure;
  name: string;
}

interface GraphPreviewMoreMenuProps {
  /** What runs: its compiled backend graph is what the JSON items export; null while it does not compile. */
  graph: WorkflowPreviewGraph | null;
  /** The workflow document the graph compiles from, which the Open-as items copy; else they convert `graph`. */
  sourceDocument?: ProjectGraphState;
  sourceId?: WorkflowInvocationSourceId;
  sourceLabel: string;
  /** Closes the dialog once the graph has opened elsewhere (the editor, a new project). */
  onClose: () => void;
}

/**
 * The preview's graph as a workflow of its own, for the More menu's Open-as items and the footer's "Edit as workflow".
 */
export const useGraphAsWorkflow = ({
  graph,
  sourceDocument,
  sourceLabel,
  onClose,
}: Pick<GraphPreviewMoreMenuProps, 'graph' | 'onClose' | 'sourceDocument' | 'sourceLabel'>) => {
  const { t } = useTranslation();
  const { workflows } = useWorkflowHostCommands();
  const graphPreview = useWorkflowGraphPreview();
  const notifications = useWorkflowNotifications();
  // A hook (not `getInvocationTemplatesSnapshot()`) so the items' disabled state updates live if the menu is opened
  // while templates are still loading.
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  // Converting a compiled graph needs the templates; copying a document does not.
  const canConvert = sourceDocument !== undefined || (graph !== null && templatesSnapshot.status === 'loaded');

  /**
   * The graph as a new workflow document of its own: a copy of the source's document where it has one, so its labels,
   * notes and form survive; else the compiled graph converted, without the node types this server lacks. Null, with
   * the reason shown, when nothing would be left.
   */
  const toNewDocument = useCallback(
    (failedKey: string): ProjectGraphState | null => {
      if (sourceDocument) {
        return {
          ...cloneProjectGraph(sourceDocument),
          id: createWorkflowId('workflow'),
          name: sourceDocument.name || sourceLabel,
        };
      }

      if (!graph) {
        return null;
      }

      const converted = previewGraphToDocument(graph, templatesSnapshot.templates);

      if (converted.document.nodes.length === 0) {
        notifications.error(t(failedKey));
        return null;
      }

      if (converted.skippedNodeTypes.length > 0) {
        notifications.info(
          t('graphPreview.nodesSkipped', {
            count: converted.skippedNodeTypes.length,
            types: converted.skippedNodeTypes.join(', '),
          })
        );
      }

      converted.document.name = graph.label ?? sourceLabel;

      return converted.document;
    },
    [graph, notifications, sourceDocument, sourceLabel, t, templatesSnapshot]
  );

  /** Adds the graph to the project as a workflow and opens it in the editor, closing the preview. */
  const editInEditor = useCallback(() => {
    const document = toNewDocument('graphPreview.editInEditorFailed');

    if (!document) {
      return;
    }

    // The preview becomes a workflow of its own beside the project's others (or takes over an untouched blank).
    workflows.addWorkflow(document, { label: t('graphPreview.openedFromPreview'), reusePlaceholder: true });
    graphPreview.openWorkflowEditor();
    onClose();
  }, [toNewDocument, t, workflows, graphPreview, onClose]);

  return { canConvert, editInEditor, toNewDocument };
};

/**
 * The graph preview's More menu: the compiled graph's JSON (what a bug report needs), and the graph opened as an editor
 * workflow, a library workflow or a project of its own.
 */
export const GraphPreviewMoreMenu = ({
  graph,
  sourceDocument,
  sourceId,
  sourceLabel,
  onClose,
}: GraphPreviewMoreMenuProps) => {
  const { t } = useTranslation();
  const menuIds = useTooltipTriggerIds();
  const session = useAuthSession();
  const { project } = useWorkflowUi();
  const graphPreview = useWorkflowGraphPreview();
  const notifications = useWorkflowNotifications();
  const { retry: retryPublication, saveDocumentAsNew } = useWorkflowPublication();
  const [saveRequest, setSaveRequest] = useState<PreviewSaveRequest | null>(null);
  const [isSavePending, setIsSavePending] = useState(false);
  const activePublicationRef = useRef<{ projectId: string; workflowId: string } | null>(null);
  const singleUserPublicationTargetsRef = useRef(new Set<{ projectId: string; workflowId: string }>());
  const isMountedRef = useRef(false);
  const canEditInEditor = sourceId !== 'workflow';

  useMountEffect(() => {
    isMountedRef.current = true;

    return () => {
      isMountedRef.current = false;
      const activePublication = activePublicationRef.current;

      if (activePublication) {
        discardUnresolvedWorkflowPublication(activePublication.projectId, activePublication.workflowId);
      }

      singleUserPublicationTargetsRef.current.forEach(({ projectId, workflowId }) => {
        discardUnresolvedWorkflowPublication(projectId, workflowId);
      });
      singleUserPublicationTargetsRef.current.clear();
    };
  });

  const {
    canConvert,
    editInEditor: handleEditInEditor,
    toNewDocument,
  } = useGraphAsWorkflow({
    graph,
    sourceDocument,
    sourceLabel,
    onClose,
  });

  const saveToLibrary = useCallback(async () => {
    const document = toNewDocument('graphPreview.saveToLibraryFailed');

    if (!document) {
      return;
    }

    if (session.multiuserEnabled) {
      setSaveRequest({ document, name: document.name });
      return;
    }

    // Single-user saves keep the backend's public-by-default behavior and retry a lost answer once.
    const activePublication = { projectId: project.getSnapshot().id, workflowId: document.id };
    singleUserPublicationTargetsRef.current.add(activePublication);
    const result = await saveDocumentAsNew(document, document.name);

    if (!isMountedRef.current) {
      discardUnresolvedWorkflowPublication(activePublication.projectId, activePublication.workflowId);
      singleUserPublicationTargetsRef.current.delete(activePublication);
      return;
    }

    if (result.status === 'failed') {
      const retried = await result.retry();

      if (!isMountedRef.current) {
        discardUnresolvedWorkflowPublication(activePublication.projectId, activePublication.workflowId);
        singleUserPublicationTargetsRef.current.delete(activePublication);
        return;
      }

      if (retried.status === 'failed') {
        notifications.error(t('graphPreview.saveToLibraryFailed'), retried.message);
      } else {
        singleUserPublicationTargetsRef.current.delete(activePublication);
      }
    } else {
      singleUserPublicationTargetsRef.current.delete(activePublication);
    }
  }, [toNewDocument, saveDocumentAsNew, notifications, project, session.multiuserEnabled, t]);
  const handleSaveToLibrary = useCallback(() => void saveToLibrary(), [saveToLibrary]);

  const closeSaveDialog = useCallback(() => {
    const activePublication = activePublicationRef.current;

    if (saveRequest?.failure && activePublication) {
      discardUnresolvedWorkflowPublication(activePublication.projectId, activePublication.workflowId);
    }

    activePublicationRef.current = null;
    setSaveRequest(null);
  }, [saveRequest]);

  const handleSaveSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!saveRequest) {
        return;
      }

      const formData = new FormData(event.currentTarget);
      const name = formData.get('templateName')?.toString().trim() || t('workflowLibrary.untitled');
      const isPublic = formData.has('isPublic');
      const activePublication = { projectId: project.getSnapshot().id, workflowId: saveRequest.document.id };

      activePublicationRef.current = activePublication;
      setIsSavePending(true);
      try {
        const result = await saveDocumentAsNew(saveRequest.document, name, isPublic);

        if (!isMountedRef.current) {
          discardUnresolvedWorkflowPublication(activePublication.projectId, activePublication.workflowId);
          return;
        }

        if (result.status === 'failed' || result.status === 'partial') {
          setSaveRequest({ ...saveRequest, failure: result, name });
        } else {
          activePublicationRef.current = null;
          setSaveRequest(null);
        }
      } finally {
        if (isMountedRef.current) {
          setIsSavePending(false);
        }
      }
    },
    [project, saveDocumentAsNew, saveRequest, t]
  );

  const handleSaveRetry = useCallback(async () => {
    if (!saveRequest?.failure) {
      return;
    }

    const activePublication = activePublicationRef.current;
    setIsSavePending(true);
    try {
      const result = await retryPublication(saveRequest.failure);

      if (!isMountedRef.current) {
        if (activePublication) {
          discardUnresolvedWorkflowPublication(activePublication.projectId, activePublication.workflowId);
        }
        return;
      }

      if (result.status === 'failed' || result.status === 'partial') {
        setSaveRequest({ ...saveRequest, failure: result });
      } else {
        activePublicationRef.current = null;
        setSaveRequest(null);
      }
    } finally {
      if (isMountedRef.current) {
        setIsSavePending(false);
      }
    }
  }, [retryPublication, saveRequest]);

  const handleSaveDialogOpenChange = useCallback(
    (event: { open: boolean }) => (event.open ? undefined : closeSaveDialog()),
    [closeSaveDialog]
  );

  const handleForkIntoProject = useCallback(() => {
    const document = toNewDocument('graphPreview.forkIntoProjectFailed');

    if (!document) {
      return;
    }

    // The port creates + activates a fresh project before loading the
    // document, so the project this preview came from is left untouched.
    graphPreview.openDocumentInNewProject(document, t('graphPreview.openedFromPreview'));
    onClose();
  }, [toNewDocument, t, graphPreview, onClose]);

  const handleCopyJson = useCallback(() => {
    if (!graph) {
      return;
    }

    copyText(getGraphJsonText(graph))
      .then(() => notifications.success(t('graphPreview.graphJsonCopied')))
      .catch(() => notifications.error(t('graphPreview.copyFailed')));
  }, [graph, notifications, t]);
  const handleDownloadJson = useCallback(() => {
    if (!graph) {
      return;
    }

    const fileName = `${(graph.label ?? 'graph').trim().replaceAll(/\s+/g, '-').toLowerCase()}.json`;
    downloadText(getGraphJsonText(graph), fileName, 'application/json');
  }, [graph]);
  // Nothing compiled, nothing to export: the menu says why rather than hiding the items.
  const jsonUnavailable = !graph ? t('graphPreview.invalidTitle') : undefined;

  return (
    <>
      <Menu.Root ids={menuIds}>
        {/* Inside a dialog the tooltip must sit inside the menu trigger: wrapped the other way round, the menu never
            takes focus and the first pointer move onto it closes it. */}
        <Menu.Trigger asChild>
          <Tooltip content={t('graphPreview.more')} ids={menuIds}>
            <IconButton aria-label={t('graphPreview.more')} data-preview-more="" size="sm" variant="outline">
              <EllipsisIcon />
            </IconButton>
          </Tooltip>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="min(14rem, var(--available-width))" maxW="var(--available-width)">
              <MenuActionItem
                icon={CopyIcon}
                label={t('graphPreview.copyGraphJson')}
                unavailableReason={jsonUnavailable ?? (canCopyText() ? undefined : t('common.copyUnavailable'))}
                value="copy-graph-json"
                onSelect={handleCopyJson}
              />
              <MenuActionItem
                icon={DownloadIcon}
                label={t('graphPreview.downloadGraphJson')}
                unavailableReason={jsonUnavailable}
                value="download-graph-json"
                onSelect={handleDownloadJson}
              />
              <Menu.Separator borderColor="border.subtle" />
              {canEditInEditor ? (
                <MenuActionItem
                  disabled={!canConvert}
                  icon={PencilRulerIcon}
                  label={t('graphPreview.editInEditor')}
                  value="edit-in-editor"
                  onSelect={handleEditInEditor}
                />
              ) : null}
              <MenuActionItem
                disabled={!canConvert}
                icon={BookmarkIcon}
                label={t('graphPreview.saveToLibrary')}
                value="save-to-library"
                onSelect={handleSaveToLibrary}
              />
              <MenuActionItem
                disabled={!canConvert}
                icon={GitForkIcon}
                label={t('graphPreview.forkIntoProject')}
                value="fork-into-project"
                onSelect={handleForkIntoProject}
              />
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <Dialog.Root
        closeOnEscape={!isSavePending}
        closeOnInteractOutside={!isSavePending}
        lazyMount
        open={saveRequest !== null}
        size="xs"
        unmountOnExit
        onOpenChange={handleSaveDialogOpenChange}
      >
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content data-preview-save-to-library-dialog>
              <chakra.form onSubmit={handleSaveSubmit}>
                <Dialog.Header>
                  <Dialog.Title>{t('workflowLibrary.saveToLibraryTitle')}</Dialog.Title>
                </Dialog.Header>
                <Dialog.Body>
                  {saveRequest?.failure ? (
                    <Text color="fg.error" fontSize="lg" role="alert">
                      {saveRequest.failure.status === 'partial'
                        ? t('workflowLibrary.publicVisibilityPartial', { message: saveRequest.failure.message })
                        : t('workflowLibrary.retryBody', {
                            message: saveRequest.failure.message,
                            name: saveRequest.name,
                          })}
                    </Text>
                  ) : (
                    <Stack gap="3">
                      <Field label={t('workflowLibrary.templateName')}>
                        <Input defaultValue={saveRequest?.name} name="templateName" size="lg" />
                      </Field>
                      <Text color="fg.muted" fontSize="md">
                        {t('workflowLibrary.saveToLibraryExplanation')}
                      </Text>
                      <Checkbox.Root colorPalette="accent" size="sm">
                        <Checkbox.HiddenInput name="isPublic" />
                        <Checkbox.Control />
                        <Checkbox.Label>
                          <Stack gap="0">
                            <Text>{t('workflowLibrary.publicWorkflow')}</Text>
                            <Text color="fg.subtle" fontSize="xs">
                              {t('workflowLibrary.publicVisibilityHelp')}
                            </Text>
                          </Stack>
                        </Checkbox.Label>
                      </Checkbox.Root>
                    </Stack>
                  )}
                </Dialog.Body>
                <Dialog.Footer>
                  <Button disabled={isSavePending} type="button" variant="ghost" onClick={closeSaveDialog}>
                    {t('common.cancel')}
                  </Button>
                  {saveRequest?.failure ? (
                    <Button loading={isSavePending} type="button" variant="solid" onClick={handleSaveRetry}>
                      {t('workflowLibrary.retry')}
                    </Button>
                  ) : (
                    <Button loading={isSavePending} type="submit" variant="solid">
                      {t('workflowLibrary.saveToLibraryConfirm')}
                    </Button>
                  )}
                </Dialog.Footer>
                <Dialog.CloseTrigger asChild>
                  <CloseButton disabled={isSavePending} />
                </Dialog.CloseTrigger>
              </chakra.form>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    </>
  );
};
