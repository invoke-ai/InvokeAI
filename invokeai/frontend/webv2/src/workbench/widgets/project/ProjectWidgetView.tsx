import type { Project } from '@workbench/projectContracts';

import { HStack, Input, Stack, Text } from '@chakra-ui/react';
import { IconButton, Field, FieldLabel, Panel } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useProjectSyncSelector } from '@workbench/projects/syncStore';
import { useNotify } from '@workbench/useNotify';
import {
  shallowEqual,
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { CopyIcon } from 'lucide-react';
import { useCallback, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

export const ProjectWidgetView = () => {
  const activeProject = useActiveProjectSelector(
    (project) => ({
      events: project.events,
      id: project.id,
      name: project.name,
      queue: project.queue,
      workflows: project.workflows,
    }),
    shallowEqual
  );

  return (
    <Stack gap="5" p="3">
      <NameSection project={activeProject} />
      <DetailsSection project={activeProject} />
    </Stack>
  );
};

type ProjectPanelViewModel = Pick<Project, 'events' | 'id' | 'name' | 'queue' | 'workflows'>;

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

const formatTimestamp = (timestamp: string | undefined, unknownTime: string): string => {
  if (!timestamp) {
    return unknownTime;
  }

  const date = new Date(timestamp);

  return Number.isNaN(date.getTime()) ? unknownTime : date.toLocaleString();
};

const DetailsSection = ({ project }: { project: ProjectPanelViewModel }) => {
  const { t } = useTranslation();
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status);
  const lastSavedAt = useWorkbenchSelector((snapshot) => snapshot.autosave.lastSavedAt);
  const projectSync = useProjectSyncSelector((snapshot) => snapshot.projects[project.id]);
  const notify = useNotify();
  let running = 0;
  let queued = 0;
  for (const item of project.queue.items) {
    if (item.status === 'running' || item.cancellationPending) {
      running += 1;
    } else if (item.status === 'pending') {
      queued += 1;
    }
  }

  const syncLabel =
    backendConnectionStatus !== 'connected'
      ? t('widgets.project.syncOffline')
      : projectSync?.schemaRefusal
        ? t('widgets.project.syncUpdateClient')
        : projectSync?.conflict
          ? t('widgets.project.syncConflict')
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
            <MiddleTruncate fontFamily="mono" fontSize="2xs" text={project.id} />
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
        <DetailRow label={t('widgets.project.graphNodes')}>
          {project.workflows.entries.reduce((count, entry) => count + entry.document.nodes.length, 0)}
        </DetailRow>
        <DetailRow label={t('widgets.project.workflows')}>{project.workflows.entries.length}</DetailRow>
        <DetailRow label={t('widgets.project.activeRuns')}>
          {t('widgets.project.activeRunCounts', { running, queued })}
        </DetailRow>
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
