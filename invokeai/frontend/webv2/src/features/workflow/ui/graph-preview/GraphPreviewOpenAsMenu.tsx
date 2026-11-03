import type { ProjectGraphState } from '@features/workflow/core/types';
import type { WorkflowInvocationSourceId, WorkflowPreviewGraph } from '@features/workflow/ui/contracts';

import { Checkbox, chakra, Input, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { useAuthSession } from '@features/identity';
import { previewGraphToDocument } from '@features/workflow/core/graphToDocument';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import {
  discardUnresolvedWorkflowPublication,
  useWorkflowPublication,
  type WorkflowPublicationFailure,
} from '@features/workflow/ui/library/useWorkflowPublication';
import { MenuActionItem } from '@features/workflow/ui/MenuActionItem';
import {
  useWorkflowGraphPreview,
  useWorkflowHostCommands,
  useWorkflowNotifications,
  useWorkflowUi,
} from '@features/workflow/ui/WorkflowUiContext';
import { downloadText } from '@platform/browser/downloadBlob';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { Field } from '@platform/ui/Field';
import { MenuContent } from '@platform/ui/Menu';
import { BookmarkIcon, DownloadIcon, GitForkIcon, PencilRulerIcon } from 'lucide-react';
import { useCallback, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

const MENU_POSITIONING = { placement: 'top-end' } as const;

interface PreviewSaveRequest {
  document: ProjectGraphState;
  failure?: WorkflowPublicationFailure;
  name: string;
}

interface GraphPreviewOpenAsMenuProps {
  children: ReactNode;
  graph: WorkflowPreviewGraph;
  sourceId?: WorkflowInvocationSourceId;
  sourceLabel: string;
  /** Dialog close after a successful "edit in editor". */
  onClose: () => void;
}

/**
 * Convert previews into editor/library/project documents or download raw JSON. Keep the trigger in the dialog to
 * avoid another platform-barrel dependency.
 */
export const GraphPreviewOpenAsMenu = ({
  children,
  graph,
  sourceId,
  sourceLabel,
  onClose,
}: GraphPreviewOpenAsMenuProps) => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const { workflows } = useWorkflowHostCommands();
  const { project } = useWorkflowUi();
  const graphPreview = useWorkflowGraphPreview();
  const notifications = useWorkflowNotifications();
  const { retry: retryPublication, saveDocumentAsNew } = useWorkflowPublication();
  const [saveRequest, setSaveRequest] = useState<PreviewSaveRequest | null>(null);
  const [isSavePending, setIsSavePending] = useState(false);
  const activePublicationRef = useRef<{ projectId: string; workflowId: string } | null>(null);
  const singleUserPublicationTargetsRef = useRef(new Set<{ projectId: string; workflowId: string }>());
  const isMountedRef = useRef(false);
  // A hook (not `getInvocationTemplatesSnapshot()`) so the "Edit in workflow
  // editor" item's disabled state updates live if the menu is opened while
  // templates are still loading.
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const canEditInEditor = sourceId !== 'workflow';
  const isTemplatesLoaded = templatesSnapshot.status === 'loaded';

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

  const handleEditInEditor = useCallback(() => {
    const { document, skippedNodeTypes } = previewGraphToDocument(graph, templatesSnapshot.templates);

    if (document.nodes.length === 0) {
      notifications.error(t('graphPreview.editInEditorFailed'));
      return;
    }

    if (skippedNodeTypes.length > 0) {
      notifications.info(
        t('graphPreview.nodesSkipped', { count: skippedNodeTypes.length, types: skippedNodeTypes.join(', ') })
      );
    }

    // The preview becomes a workflow of its own beside the project's others (or takes over an untouched blank).
    workflows.addWorkflow(document, { label: t('graphPreview.openedFromPreview'), reusePlaceholder: true });
    graphPreview.openWorkflowEditor();
    onClose();
  }, [graph, templatesSnapshot, notifications, t, workflows, graphPreview, onClose]);

  const saveToLibrary = useCallback(async () => {
    const { document, skippedNodeTypes } = previewGraphToDocument(graph, templatesSnapshot.templates);

    if (document.nodes.length === 0) {
      notifications.error(t('graphPreview.saveToLibraryFailed'));
      return;
    }

    if (skippedNodeTypes.length > 0) {
      notifications.info(
        t('graphPreview.nodesSkipped', { count: skippedNodeTypes.length, types: skippedNodeTypes.join(', ') })
      );
    }

    document.name = graph.label ?? sourceLabel;

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
  }, [graph, templatesSnapshot, sourceLabel, saveDocumentAsNew, notifications, project, session.multiuserEnabled, t]);
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
    const { document, skippedNodeTypes } = previewGraphToDocument(graph, templatesSnapshot.templates);

    if (document.nodes.length === 0) {
      notifications.error(t('graphPreview.forkIntoProjectFailed'));
      return;
    }

    if (skippedNodeTypes.length > 0) {
      notifications.info(
        t('graphPreview.nodesSkipped', { count: skippedNodeTypes.length, types: skippedNodeTypes.join(', ') })
      );
    }

    document.name = graph.label ?? sourceLabel;

    // The port creates + activates a fresh project before loading the
    // document, so the project this preview came from is left untouched.
    graphPreview.openDocumentInNewProject(document, t('graphPreview.openedFromPreview'));
    onClose();
  }, [graph, templatesSnapshot, sourceLabel, notifications, t, graphPreview, onClose]);

  const handleDownloadJson = useCallback(() => {
    const fileName = `${(graph.label ?? 'graph').trim().replaceAll(/\s+/g, '-').toLowerCase()}.json`;
    downloadText(JSON.stringify(graph.backendGraph ?? graph, null, 2), fileName, 'application/json');
  }, [graph]);

  return (
    <>
      <Menu.Root positioning={MENU_POSITIONING}>
        <Menu.Trigger asChild>{children}</Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="16rem">
              {canEditInEditor ? (
                <MenuActionItem
                  hint={t('graphPreview.editInEditorHint')}
                  icon={PencilRulerIcon}
                  isDisabled={!isTemplatesLoaded}
                  label={t('graphPreview.editInEditor')}
                  value="edit-in-editor"
                  onSelect={handleEditInEditor}
                />
              ) : null}
              <MenuActionItem
                hint={t('graphPreview.saveToLibraryHint')}
                icon={BookmarkIcon}
                isDisabled={!isTemplatesLoaded}
                label={t('graphPreview.saveToLibrary')}
                value="save-to-library"
                onSelect={handleSaveToLibrary}
              />
              <MenuActionItem
                hint={t('graphPreview.forkIntoProjectHint')}
                icon={GitForkIcon}
                isDisabled={!isTemplatesLoaded}
                label={t('graphPreview.forkIntoProject')}
                value="fork-into-project"
                onSelect={handleForkIntoProject}
              />
              <MenuActionItem
                hint={t('graphPreview.downloadJsonHint')}
                icon={DownloadIcon}
                label={t('graphPreview.downloadJson')}
                value="download-json"
                onSelect={handleDownloadJson}
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
