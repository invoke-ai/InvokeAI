import type { Project } from '@workbench/projectContracts';

import { HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { flushGenerateDrafts } from '@features/generation/drafts';
import { Button, IconButton, ConfirmDialog, Field, FieldLabel, Panel } from '@platform/ui';
import { useProjectSyncSelector } from '@workbench/projects/syncStore';
import { useProjectActions } from '@workbench/projects/useProjectActions';
import { useNotify } from '@workbench/useNotify';
import {
  shallowEqual,
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { ArrowRightIcon, CopyIcon, History as HistoryIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

const RECOVERY_DELETE_HOVER = { color: 'fg.error' } as const;

/**
 * The Project panel: rename the active project, see its sync/debug details,
 * and manage recovery copies (open or delete them). Recoveries are keyed to
 * their root original via `recoveryOf`, so the whole family is visible from
 * any of its members.
 */
export const ProjectWidgetView = () => {
  const activeProject = useActiveProjectSelector(
    (project) => ({
      events: project.events,
      graphHistory: project.graphHistory,
      id: project.id,
      name: project.name,
      projectGraph: project.projectGraph,
      queue: project.queue,
      recoveredAt: project.recoveredAt,
      recoveryOf: project.recoveryOf,
    }),
    shallowEqual
  );

  return (
    <Stack gap="5" p="3">
      <NameSection project={activeProject} />
      <RecoverySection project={activeProject} />
      <DetailsSection project={activeProject} />
    </Stack>
  );
};

type ProjectPanelViewModel = Pick<
  Project,
  'events' | 'graphHistory' | 'id' | 'name' | 'projectGraph' | 'queue' | 'recoveredAt' | 'recoveryOf'
>;

const NameSection = ({ project }: { project: ProjectPanelViewModel }) => {
  const { t } = useTranslation();
  const { projects } = useWorkbenchCommands();

  const commitName = useCallback(
    (value: string) => {
      const name = value.trim();

      if (name && name !== project.name) {
        projects.rename(project.id, name);
      }
    },
    [project.id, project.name, projects]
  );
  const handleBlur = useCallback(
    (event: React.FocusEvent<HTMLInputElement>) => commitName(event.currentTarget.value),
    [commitName]
  );
  const handleKeyDown = useCallback((event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.currentTarget.blur();
    }
  }, []);

  return (
    <Field helpText={t('widgets.project.nameHelp')} label={t('widgets.project.nameLabel')}>
      <Input
        defaultValue={project.name}
        key={`${project.id}:${project.name}`}
        size="sm"
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
      />
    </Field>
  );
};

/** The id whose recovery family this project belongs to. */
const getRecoveryRootId = (project: Pick<Project, 'id' | 'recoveryOf'>): string => project.recoveryOf ?? project.id;

const formatTimestamp = (timestamp: string | undefined, unknownTime: string): string => {
  if (!timestamp) {
    return unknownTime;
  }

  const date = new Date(timestamp);

  return Number.isNaN(date.getTime()) ? unknownTime : date.toLocaleString();
};

const RecoverySection = ({ project }: { project: ProjectPanelViewModel }) => {
  const { t } = useTranslation();
  const projects = useWorkbenchSelector((snapshot) => snapshot.projects);
  const { projects: projectCommands } = useWorkbenchCommands();
  const { deleteProject } = useProjectActions();
  const [deleteTarget, setDeleteTarget] = useState<Project | null>(null);
  const rootId = getRecoveryRootId(project);
  const original = project.recoveryOf ? projects.find((candidate) => candidate.id === project.recoveryOf) : null;
  const recoveries = projects.filter(
    (candidate) => candidate.id !== project.id && getRecoveryRootId(candidate) === rootId && candidate.recoveryOf
  );
  const handleOpenOriginal = useCallback(() => {
    if (original) {
      flushGenerateDrafts();
      projectCommands.switchTo(original.id);
    }
  }, [original, projectCommands]);
  const handleCloseDeleteDialog = useCallback(() => setDeleteTarget(null), []);
  const handleConfirmDelete = useCallback(async () => {
    if (deleteTarget) {
      await deleteProject(deleteTarget);
    }
  }, [deleteProject, deleteTarget]);

  if (!project.recoveryOf && recoveries.length === 0) {
    return null;
  }

  return (
    <Stack gap="2">
      <FieldLabel>{t('widgets.project.recoveries')}</FieldLabel>
      {project.recoveryOf ? (
        <Panel p="2.5">
          <HStack gap="2">
            <Icon as={HistoryIcon} boxSize="3.5" color="fg.muted" flexShrink={0} />
            <Stack flex="1" gap="0" minW="0">
              <Text fontSize="xs" fontWeight="600">
                {t('widgets.project.recoveryCopy')}
              </Text>
              <Text color="fg.muted" fontSize="2xs">
                {t('widgets.project.recoveryForkedDescription', {
                  time: formatTimestamp(project.recoveredAt, t('common.unknownTime')),
                })}
              </Text>
            </Stack>
          </HStack>
          {original ? (
            <Button mt="2" size="2xs" variant="outline" w="full" onClick={handleOpenOriginal}>
              <ArrowRightIcon />
              {t('widgets.project.openOriginal', { name: original.name })}
            </Button>
          ) : null}
        </Panel>
      ) : null}
      {recoveries.map((recovery) => (
        <RecoveryRow key={recovery.id} recovery={recovery} onDelete={setDeleteTarget} />
      ))}
      <ConfirmDialog
        body={t('widgets.project.deleteRecoveryBody', { name: deleteTarget?.name ?? '' })}
        confirmLabel={t('widgets.project.deleteRecovery')}
        isOpen={deleteTarget !== null}
        title={t('widgets.project.deleteRecoveryTitle')}
        onClose={handleCloseDeleteDialog}
        onConfirm={handleConfirmDelete}
      />
    </Stack>
  );
};

const RecoveryRow = ({
  recovery,
  onDelete,
}: {
  recovery: Project;
  onDelete: React.Dispatch<React.SetStateAction<Project | null>>;
}) => {
  const { t } = useTranslation();
  const { projects } = useWorkbenchCommands();
  const handleOpen = useCallback(() => {
    flushGenerateDrafts();
    projects.switchTo(recovery.id);
  }, [projects, recovery.id]);
  const handleDelete = useCallback(() => onDelete(recovery), [onDelete, recovery]);

  return (
    <Panel alignItems="center" flexDirection="row" gap="2" p="2">
      <Stack flex="1" gap="0" minW="0">
        <Text fontSize="xs" fontWeight="600" truncate>
          {recovery.name}
        </Text>
        <Text color="fg.muted" fontSize="2xs">
          {formatTimestamp(recovery.recoveredAt, t('common.unknownTime'))}
        </Text>
      </Stack>
      <IconButton
        aria-label={t('widgets.project.openRecovery', { name: recovery.name })}
        color="fg.muted"
        size="2xs"
        variant="ghost"
        onClick={handleOpen}
      >
        <ArrowRightIcon />
      </IconButton>
      <IconButton
        aria-label={t('widgets.project.deleteRecoveryAria', { name: recovery.name })}
        color="fg.muted"
        size="2xs"
        variant="ghost"
        _hover={RECOVERY_DELETE_HOVER}
        onClick={handleDelete}
      >
        <Trash2Icon />
      </IconButton>
    </Panel>
  );
};

const DetailsSection = ({ project }: { project: ProjectPanelViewModel }) => {
  const { t } = useTranslation();
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status);
  const lastSavedAt = useWorkbenchSelector((snapshot) => snapshot.autosave.lastSavedAt);
  const projectSync = useProjectSyncSelector((snapshot) => snapshot.projects[project.id]);
  const notify = useNotify();

  const syncLabel =
    backendConnectionStatus !== 'connected'
      ? t('widgets.project.syncOffline')
      : projectSync === undefined || projectSync.isPendingPush
        ? t('widgets.project.syncWaiting')
        : t('widgets.project.syncSynced', { revision: projectSync.revision ?? '—' });

  const copyId = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(project.id);
      notify.success(t('widgets.project.idCopied'));
    } catch {
      notify.error(t('common.couldNotCopy'), t('common.clipboardBlocked'));
    }
  }, [notify, project.id, t]);
  const handleCopyId = useCallback(() => void copyId(), [copyId]);

  return (
    <Stack gap="2">
      <FieldLabel>{t('common.details')}</FieldLabel>
      <Panel gap="1.5" p="2.5">
        <DetailRow label={t('common.id')}>
          <HStack gap="1" minW="0">
            <Text fontFamily="mono" fontSize="2xs" truncate>
              {project.id}
            </Text>
            <IconButton
              aria-label={t('widgets.project.copyId')}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={handleCopyId}
            >
              <CopyIcon />
            </IconButton>
          </HStack>
        </DetailRow>
        <DetailRow label={t('widgets.project.sync')}>{syncLabel}</DetailRow>
        <DetailRow label={t('common.lastSaved')}>
          {lastSavedAt ? formatTimestamp(lastSavedAt, t('common.unknownTime')) : t('common.notYet')}
        </DetailRow>
        <DetailRow label={t('widgets.project.graphNodes')}>{project.projectGraph.nodes.length}</DetailRow>
        <DetailRow label={t('widgets.project.queueItems')}>{project.queue.items.length}</DetailRow>
        <DetailRow label={t('widgets.project.events')}>{project.events.length}</DetailRow>
      </Panel>
    </Stack>
  );
};

const DetailRow = ({ children, label }: { children: ReactNode; label: string }) => (
  <HStack gap="3" justify="space-between" minH="5">
    <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
      {label}
    </Text>
    {typeof children === 'string' || typeof children === 'number' ? (
      <Text fontSize="2xs" textAlign="end">
        {children}
      </Text>
    ) : (
      children
    )}
  </HStack>
);
