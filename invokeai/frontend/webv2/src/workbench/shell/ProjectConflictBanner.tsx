import { Alert, Button, HStack, Stack } from '@chakra-ui/react';
import { downloadText } from '@platform/browser/downloadBlob';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { serializeProjectDocumentV3Json } from '@workbench/projects/projectDocument';
import { useProjectSyncSelector } from '@workbench/projects/syncStore';
import {
  useActiveProject,
  useActiveProjectId,
  useActiveProjectName,
  useWorkbenchCommands,
  useWorkbenchPersistenceAdapter,
  useWorkbenchPersistenceService,
} from '@workbench/WorkbenchContext';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

type ResolutionAction = 'delete-recoverable' | 'discard' | 'save-as-new' | 'use-server';

const getExportName = (name: string): string =>
  `${
    name
      .trim()
      .replaceAll(/[^\w.-]+/g, '-')
      .replaceAll(/^-+|-+$/g, '') || 'project-draft'
  }.json`;

export const ProjectConflictBanner = () => {
  const { t } = useTranslation();
  const projectId = useActiveProjectId();
  const project = useActiveProject();
  const projectName = useActiveProjectName();
  const sync = useProjectSyncSelector((snapshot) => snapshot.projects[projectId]);
  const localDraftStatus = useProjectSyncSelector((snapshot) => snapshot.localDraftStatus);
  const recoverableDrafts = useProjectSyncSelector((snapshot) => snapshot.recoverableDrafts);
  const persistence = useWorkbenchPersistenceService();
  const persistenceAdapter = useWorkbenchPersistenceAdapter();
  const { notifications, projects } = useWorkbenchCommands();
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [resolutionAction, setResolutionAction] = useState<ResolutionAction | null>(null);
  const [recoverableDraftIndex, setRecoverableDraftIndex] = useState(0);
  const selectedRecoverableDraftIndex = Math.min(recoverableDraftIndex, Math.max(0, recoverableDrafts.length - 1));
  const recoverableDraft = recoverableDrafts[selectedRecoverableDraftIndex];

  const run = useCallback(
    async (action: string, operation: () => Promise<void>) => {
      setPendingAction(action);
      try {
        await operation();
      } catch (error) {
        notifications.reportError({
          area: 'project-conflict',
          message: error instanceof Error ? error.message : t('shell.projectConflict.actionFailed'),
          namespace: 'system',
        });
      } finally {
        setPendingAction(null);
      }
    },
    [notifications, t]
  );

  const exportDraft = useCallback(
    () =>
      run('export', () => {
        const { documentJson } = serializeProjectDocumentV3Json(project);
        downloadText(documentJson, getExportName(projectName), 'application/json');
        return Promise.resolve();
      }),
    [project, projectName, run]
  );

  const saveAsNew = useCallback(
    () =>
      run('save-as-new', async () => {
        const result = await persistence.resolveConflictSaveAsNew(project);
        const retargeted = persistenceAdapter.retargetProject({
          boardId: result.boardId,
          name: result.name,
          project: result.project,
          projectId: result.sourceProjectId,
          sourceName: result.sourceName,
          targetProjectId: result.targetProjectId,
        });
        if (!retargeted.ok) {
          persistence.abortProjectResolution(project.id);
          throw new Error(t('shell.projectConflict.copyOpenedDuringResolution'));
        }
        persistence.acknowledgeProjectResolution(project.id);
      }),
    [persistence, persistenceAdapter, project, run, t]
  );
  const exportRecoverableDraft = useCallback(
    () =>
      run('export-recoverable', async () => {
        if (!recoverableDraft) {
          return;
        }
        const documentJson = await persistence.getRecoverableDraftDocument(
          recoverableDraft.projectId,
          recoverableDraft.editorSessionId
        );
        if (!documentJson) {
          throw new Error(t('shell.projectConflict.draftUnavailable'));
        }
        downloadText(documentJson, getExportName(recoverableDraft.projectId), 'application/json');
      }),
    [persistence, recoverableDraft, run, t]
  );
  const handleSaveAsNew = useCallback(() => setResolutionAction('save-as-new'), []);
  const handleExport = useCallback(() => void exportDraft(), [exportDraft]);
  const handleExportRecoverable = useCallback(() => void exportRecoverableDraft(), [exportRecoverableDraft]);
  const openDeleteRecoverableConfirmation = useCallback(() => setResolutionAction('delete-recoverable'), []);
  const selectPreviousRecoverableDraft = useCallback(
    () => setRecoverableDraftIndex((index) => Math.max(0, index - 1)),
    []
  );
  const selectNextRecoverableDraft = useCallback(
    () => setRecoverableDraftIndex((index) => Math.min(recoverableDrafts.length - 1, index + 1)),
    [recoverableDrafts.length]
  );
  const openUseServerConfirmation = useCallback(() => setResolutionAction('use-server'), []);
  const openDiscardConfirmation = useCallback(() => setResolutionAction('discard'), []);
  const closeConfirmation = useCallback(() => setResolutionAction(null), []);

  const confirmResolution = useCallback(async () => {
    const action = resolutionAction;
    if (action === 'delete-recoverable') {
      await run(action, async () => {
        if (!recoverableDraft) {
          return;
        }
        await persistence.deleteRecoverableDraft(
          recoverableDraft.projectId,
          recoverableDraft.editorSessionId,
          recoverableDraft.generation,
          recoverableDraft.updatedAt
        );
      });
      return;
    }
    if (action === 'save-as-new') {
      await saveAsNew();
      return;
    }
    if (action === 'use-server') {
      await run(action, async () => {
        const result = await persistence.resolveConflictUseServer(projectId);
        if (result.status !== 'loaded') {
          throw new Error(t('shell.projectConflict.serverVersionUnavailable'));
        }
        persistenceAdapter.replaceProjectFromServer({ project: result.project, projectId });
        persistence.acknowledgeProjectResolution(projectId);
      });
      return;
    }
    if (action === 'discard') {
      await run(action, async () => {
        await persistence.resolveConflictDiscard(projectId);
        try {
          let closeResult = projects.close(projectId);
          if (!closeResult.ok && closeResult.reason === 'last-project') {
            projects.create();
            closeResult = projects.close(projectId);
          }
          if (!closeResult.ok) {
            throw new Error(t('shell.projectConflict.actionFailed'));
          }
          persistence.acknowledgeProjectResolution(projectId);
        } catch (error) {
          persistence.abortProjectResolution(projectId);
          throw error;
        }
      });
    }
  }, [persistence, persistenceAdapter, projectId, projects, recoverableDraft, resolutionAction, run, saveAsNew, t]);

  const conflict = sync?.conflict;
  const schemaRefusal = sync?.schemaRefusal;
  const hasProjectAlert = Boolean(conflict || schemaRefusal || localDraftStatus !== 'ok');
  if (!hasProjectAlert && !recoverableDraft) {
    return null;
  }

  const title = schemaRefusal
    ? t('shell.projectConflict.schemaTitle')
    : conflict?.kind === 'deleted'
      ? t('shell.projectConflict.deletedTitle')
      : conflict
        ? t('shell.projectConflict.revisionTitle')
        : t('shell.projectConflict.backupTitle');
  const description = schemaRefusal
    ? t('shell.projectConflict.schemaDescription')
    : conflict?.kind === 'deleted'
      ? t('shell.projectConflict.deletedDescription')
      : conflict
        ? t('shell.projectConflict.revisionDescription')
        : t('shell.projectConflict.backupDescription');

  return (
    <>
      {recoverableDraft ? (
        <Alert.Root borderRadius="none" status="warning" variant="surface">
          <Alert.Indicator />
          <Alert.Content>
            <HStack align="center" gap="4" justify="space-between" w="full">
              <Stack gap="0.5">
                <Alert.Title>{t('shell.projectConflict.recoverableDraftTitle')}</Alert.Title>
                <Alert.Description>
                  {t('shell.projectConflict.recoverableDraftDescription', { projectId: recoverableDraft.projectId })}
                  {recoverableDrafts.length > 1
                    ? ` ${t('shell.projectConflict.recoverableDraftPosition', {
                        count: recoverableDrafts.length,
                        number: selectedRecoverableDraftIndex + 1,
                      })}`
                    : null}
                </Alert.Description>
              </Stack>
              <HStack flexShrink="0">
                {recoverableDrafts.length > 1 ? (
                  <>
                    <Button
                      disabled={pendingAction !== null || selectedRecoverableDraftIndex === 0}
                      onClick={selectPreviousRecoverableDraft}
                      size="sm"
                      variant="ghost"
                    >
                      {t('shell.projectConflict.previousDraft')}
                    </Button>
                    <Button
                      disabled={
                        pendingAction !== null || selectedRecoverableDraftIndex === recoverableDrafts.length - 1
                      }
                      onClick={selectNextRecoverableDraft}
                      size="sm"
                      variant="ghost"
                    >
                      {t('shell.projectConflict.nextDraft')}
                    </Button>
                  </>
                ) : null}
                <Button
                  disabled={pendingAction !== null}
                  loading={pendingAction === 'export-recoverable'}
                  onClick={handleExportRecoverable}
                  size="sm"
                  variant="outline"
                >
                  {t('shell.projectConflict.export')}
                </Button>
                <Button
                  disabled={pendingAction !== null}
                  onClick={openDeleteRecoverableConfirmation}
                  size="sm"
                  variant="ghost"
                >
                  {t('shell.projectConflict.deleteDraft')}
                </Button>
              </HStack>
            </HStack>
          </Alert.Content>
        </Alert.Root>
      ) : null}
      {hasProjectAlert ? (
        <Alert.Root borderRadius="none" status={conflict || schemaRefusal ? 'warning' : 'info'} variant="surface">
          <Alert.Indicator />
          <Alert.Content>
            <HStack align="center" gap="4" justify="space-between" w="full">
              <Stack gap="0.5">
                <Alert.Title>{title}</Alert.Title>
                <Alert.Description>{description}</Alert.Description>
              </Stack>
              <HStack flexShrink="0">
                {conflict ? (
                  <Button
                    disabled={pendingAction !== null}
                    loading={pendingAction === 'save-as-new'}
                    onClick={handleSaveAsNew}
                    size="sm"
                  >
                    {t('shell.projectConflict.saveAsNew')}
                  </Button>
                ) : null}
                {conflict?.kind === 'revision' ? (
                  <Button
                    disabled={pendingAction !== null}
                    onClick={openUseServerConfirmation}
                    size="sm"
                    variant="outline"
                  >
                    {t('shell.projectConflict.useServer')}
                  </Button>
                ) : null}
                {conflict?.kind === 'deleted' || schemaRefusal ? (
                  <Button
                    disabled={pendingAction !== null}
                    onClick={openDiscardConfirmation}
                    size="sm"
                    variant="outline"
                  >
                    {t('shell.projectConflict.discard')}
                  </Button>
                ) : null}
                {conflict || schemaRefusal ? (
                  <Button
                    disabled={pendingAction !== null}
                    loading={pendingAction === 'export'}
                    onClick={handleExport}
                    size="sm"
                    variant="ghost"
                  >
                    {t('shell.projectConflict.export')}
                  </Button>
                ) : null}
              </HStack>
            </HStack>
          </Alert.Content>
        </Alert.Root>
      ) : null}
      {hasProjectAlert || recoverableDraft ? (
        <ConfirmDialog
          body={t(
            resolutionAction === 'delete-recoverable'
              ? 'shell.projectConflict.deleteDraftConfirmBody'
              : resolutionAction === 'save-as-new'
                ? 'shell.projectConflict.saveAsNewConfirmBody'
                : resolutionAction === 'use-server'
                  ? 'shell.projectConflict.useServerConfirmBody'
                  : 'shell.projectConflict.discardConfirmBody'
          )}
          confirmLabel={t(
            resolutionAction === 'delete-recoverable'
              ? 'shell.projectConflict.deleteDraft'
              : resolutionAction === 'save-as-new'
                ? 'shell.projectConflict.saveAsNew'
                : resolutionAction === 'use-server'
                  ? 'shell.projectConflict.useServer'
                  : 'shell.projectConflict.discard'
          )}
          isDestructive={resolutionAction !== 'save-as-new'}
          isOpen={resolutionAction !== null}
          title={t('shell.projectConflict.confirmTitle')}
          onClose={closeConfirmation}
          onConfirm={confirmResolution}
        />
      ) : null}
    </>
  );
};
