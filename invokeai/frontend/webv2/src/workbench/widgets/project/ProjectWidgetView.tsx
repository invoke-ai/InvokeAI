import { HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { useState, type ReactNode } from 'react';
import { ArrowRightIcon, CopyIcon, History as HistoryIcon, Trash2Icon } from 'lucide-react';

import { Button, IconButton } from '../../components/ui/Button';
import { ConfirmDialog } from '../../components/ui/ConfirmDialog';
import { Field, FieldLabel } from '../../components/ui/Field';
import { Panel } from '../../components/ui/Panel';
import { useProjectSync } from '../../projects/syncStore';
import { useProjectActions } from '../../projects/useProjectActions';
import type { Project } from '../../types';
import { useNotify } from '../../useNotify';
import { useActiveProject, useWorkbenchDispatch, useWorkbenchSelector } from '../../WorkbenchContext';

/**
 * The Project panel: rename the active project, see its sync/debug details,
 * and manage recovery copies (open or delete them). Recoveries are keyed to
 * their root original via `recoveryOf`, so the whole family is visible from
 * any of its members.
 */
export const ProjectWidgetView = () => {
  const activeProject = useActiveProject();

  return (
    <Stack gap="5" p="3">
      <NameSection project={activeProject} />
      <RecoverySection project={activeProject} />
      <DetailsSection project={activeProject} />
    </Stack>
  );
};

const NameSection = ({ project }: { project: Project }) => {
  const dispatch = useWorkbenchDispatch();

  const commitName = (value: string) => {
    const name = value.trim();

    if (name && name !== project.name) {
      dispatch({ name, projectId: project.id, type: 'renameProject' });
    }
  };

  return (
    <Field helpText="Saved with the project." label="Project name">
      <Input
        defaultValue={project.name}
        key={`${project.id}:${project.name}`}
        size="sm"
        onBlur={(event) => commitName(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.currentTarget.blur();
          }
        }}
      />
    </Field>
  );
};

/** The id whose recovery family this project belongs to. */
const getRecoveryRootId = (project: Project): string => project.recoveryOf ?? project.id;

const formatTimestamp = (timestamp: string | undefined): string => {
  if (!timestamp) {
    return 'Unknown time';
  }

  const date = new Date(timestamp);

  return Number.isNaN(date.getTime()) ? 'Unknown time' : date.toLocaleString();
};

const RecoverySection = ({ project }: { project: Project }) => {
  const projects = useWorkbenchSelector((snapshot) => snapshot.state.projects);
  const dispatch = useWorkbenchDispatch();
  const { deleteProject } = useProjectActions();
  const [deleteTarget, setDeleteTarget] = useState<Project | null>(null);
  const rootId = getRecoveryRootId(project);
  const original = project.recoveryOf ? projects.find((candidate) => candidate.id === project.recoveryOf) : null;
  const recoveries = projects.filter(
    (candidate) => candidate.id !== project.id && getRecoveryRootId(candidate) === rootId && candidate.recoveryOf
  );

  if (!project.recoveryOf && recoveries.length === 0) {
    return null;
  }

  return (
    <Stack gap="2">
      <FieldLabel>Recoveries</FieldLabel>
      {project.recoveryOf ? (
        <Panel p="2.5">
          <HStack gap="2">
            <Icon as={HistoryIcon} boxSize="3.5" color="fg.muted" flexShrink={0} />
            <Stack flex="1" gap="0" minW="0">
              <Text fontSize="xs" fontWeight="600">
                Recovery copy
              </Text>
              <Text color="fg.muted" fontSize="2xs">
                Forked {formatTimestamp(project.recoveredAt)} after this project was changed elsewhere.
              </Text>
            </Stack>
          </HStack>
          {original ? (
            <Button
              mt="2"
              size="2xs"
              variant="outline"
              w="full"
              onClick={() => dispatch({ projectId: original.id, type: 'switchProject' })}
            >
              <ArrowRightIcon />
              Open original "{original.name}"
            </Button>
          ) : null}
        </Panel>
      ) : null}
      {recoveries.map((recovery) => (
        <Panel key={recovery.id} alignItems="center" flexDirection="row" gap="2" p="2">
          <Stack flex="1" gap="0" minW="0">
            <Text fontSize="xs" fontWeight="600" truncate>
              {recovery.name}
            </Text>
            <Text color="fg.muted" fontSize="2xs">
              {formatTimestamp(recovery.recoveredAt)}
            </Text>
          </Stack>
          <IconButton
            aria-label={`Open ${recovery.name}`}
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={() => dispatch({ projectId: recovery.id, type: 'switchProject' })}
          >
            <ArrowRightIcon />
          </IconButton>
          <IconButton
            aria-label={`Delete ${recovery.name}`}
            color="fg.muted"
            size="2xs"
            variant="ghost"
            _hover={{ color: 'fg.error' }}
            onClick={() => setDeleteTarget(recovery)}
          >
            <Trash2Icon />
          </IconButton>
        </Panel>
      ))}
      <ConfirmDialog
        body={`Delete "${deleteTarget?.name ?? ''}"? The recovery copy is removed permanently.`}
        confirmLabel="Delete recovery"
        isOpen={deleteTarget !== null}
        title="Delete recovery?"
        onClose={() => setDeleteTarget(null)}
        onConfirm={async () => {
          if (deleteTarget) {
            await deleteProject(deleteTarget);
          }
        }}
      />
    </Stack>
  );
};

const DetailsSection = ({ project }: { project: Project }) => {
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.state.backendConnection.status);
  const lastSavedAt = useWorkbenchSelector((snapshot) => snapshot.state.autosave.lastSavedAt);
  const sync = useProjectSync();
  const notify = useNotify();
  const projectSync = sync.projects[project.id];

  const syncLabel =
    backendConnectionStatus !== 'connected'
      ? 'Offline — saved in this browser'
      : projectSync === undefined || projectSync.isPendingPush
        ? 'Waiting to sync'
        : `Synced (revision ${projectSync.revision ?? '—'})`;

  const copyId = async () => {
    try {
      await navigator.clipboard.writeText(project.id);
      notify.success('Project id copied');
    } catch {
      notify.error('Could not copy', 'Clipboard access was blocked by the browser.');
    }
  };

  return (
    <Stack gap="2">
      <FieldLabel>Details</FieldLabel>
      <Panel gap="1.5" p="2.5">
        <DetailRow label="ID">
          <HStack gap="1" minW="0">
            <Text fontFamily="mono" fontSize="2xs" truncate>
              {project.id}
            </Text>
            <IconButton
              aria-label="Copy project id"
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={() => void copyId()}
            >
              <CopyIcon />
            </IconButton>
          </HStack>
        </DetailRow>
        <DetailRow label="Sync">{syncLabel}</DetailRow>
        <DetailRow label="Last saved">{lastSavedAt ? formatTimestamp(lastSavedAt) : 'Not yet'}</DetailRow>
        <DetailRow label="Graph nodes">{project.projectGraph.nodes.length}</DetailRow>
        <DetailRow label="Queue items">{project.queue.items.length}</DetailRow>
        <DetailRow label="Events">{project.events.length}</DetailRow>
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
