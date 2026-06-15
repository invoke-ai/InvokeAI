import type { ModelInstallJob, ModelInstallStatus } from '@workbench/models/types';

import { Badge, Box, Flex, HStack, Icon, Progress, Spinner, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton, Scrollable, Tooltip } from '@workbench/components/ui';
import {
  cancelModelInstall,
  pauseModelInstall,
  pruneCompletedModelInstalls,
  restartFailedModelInstall,
  resumeModelInstall,
} from '@workbench/models/api';
import {
  ensureInstallsLoaded,
  isActiveInstallStatus,
  refreshInstalls,
  replaceInstallJob,
  useInstallProgress,
  useInstallsSnapshot,
} from '@workbench/models/installsStore';
import { formatBytes, getInstallSourceLabel } from '@workbench/models/taxonomy';
import { useNotify } from '@workbench/useNotify';
import { PauseIcon, PlayIcon, RotateCcwIcon, Trash2Icon, XIcon } from 'lucide-react';
import { useEffect, useState } from 'react';

const STATUS_BADGES: Record<ModelInstallStatus, { label: string; palette: string }> = {
  cancelled: { label: 'Cancelled', palette: 'gray' },
  completed: { label: 'Installed', palette: 'green' },
  downloading: { label: 'Downloading', palette: 'blue' },
  downloads_done: { label: 'Downloaded', palette: 'blue' },
  error: { label: 'Failed', palette: 'red' },
  paused: { label: 'Paused', palette: 'orange' },
  running: { label: 'Installing', palette: 'blue' },
  waiting: { label: 'Waiting', palette: 'gray' },
};

/**
 * The model install/download queue: live rows for every install job with
 * pause/resume/cancel/retry per job and pruning of settled jobs. Download
 * progress streams in over the socket and bypasses the job list re-render.
 */
export const InstallQueueSection = () => {
  const notify = useNotify();
  const { error, jobs, status } = useInstallsSnapshot();
  const [isPruning, setIsPruning] = useState(false);

  useEffect(() => {
    ensureInstallsLoaded();
  }, []);

  const settledCount = jobs.filter((job) => !isActiveInstallStatus(job.status) && job.status !== 'paused').length;

  const handlePrune = async () => {
    setIsPruning(true);

    try {
      await pruneCompletedModelInstalls();
      await refreshInstalls();
    } catch (pruneError) {
      notify.error('Prune failed', pruneError instanceof Error ? pruneError.message : String(pruneError));
    } finally {
      setIsPruning(false);
    }
  };

  if (status === 'loading' || status === 'idle') {
    return (
      <Flex align="center" justify="center" py="6">
        <Spinner color="fg.subtle" size="sm" />
      </Flex>
    );
  }

  if (status === 'error') {
    return (
      <Stack align="center" gap="1" py="6">
        <Text color="fg.error" fontSize="xs" fontWeight="600">
          Could not load the install queue
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {error}
        </Text>
        <Button mt="1" size="xs" variant="outline" onClick={() => void refreshInstalls()}>
          Retry
        </Button>
      </Stack>
    );
  }

  if (jobs.length === 0) {
    return (
      <Stack align="center" gap="1" py="6">
        <Text color="fg.muted" fontSize="xs" fontWeight="600">
          No installs yet
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          Models you install will download here.
        </Text>
      </Stack>
    );
  }

  return (
    <Stack gap="2" h="full" minH="0">
      <HStack justify="space-between">
        <Text color="fg.subtle" fontSize="2xs">
          {jobs.length} job{jobs.length === 1 ? '' : 's'}
        </Text>
        <Button
          disabled={settledCount === 0}
          loading={isPruning}
          size="2xs"
          variant="ghost"
          onClick={() => void handlePrune()}
        >
          <Icon as={Trash2Icon} boxSize="3" />
          Clear finished
        </Button>
      </HStack>
      <Scrollable flex="1" label="Install jobs" minH="0">
        <Stack gap="1.5">
          {jobs.map((job) => (
            <InstallJobRow key={job.id} job={job} onError={notify.error} />
          ))}
        </Stack>
      </Scrollable>
    </Stack>
  );
};

const InstallJobRow = ({
  job,
  onError,
}: {
  job: ModelInstallJob;
  onError: (title: string, message: string) => void;
}) => {
  const [isActing, setIsActing] = useState(false);
  const badge = STATUS_BADGES[job.status] ?? { label: job.status, palette: 'gray' };
  const sourceLabel = getInstallSourceLabel(job.source);
  const displayName = job.config_out?.name ?? sourceLabel;

  const runAction = async (action: () => Promise<unknown>, failureTitle: string) => {
    setIsActing(true);

    try {
      const result = await action();

      if (result && typeof result === 'object' && 'id' in (result as ModelInstallJob)) {
        replaceInstallJob(result as ModelInstallJob);
      } else {
        await refreshInstalls();
      }
    } catch (actionError) {
      onError(failureTitle, actionError instanceof Error ? actionError.message : String(actionError));
    } finally {
      setIsActing(false);
    }
  };

  return (
    <Stack
      bg="bg.subtle"
      borderColor={job.status === 'error' ? 'border.error' : 'border.subtle'}
      borderWidth="1px"
      gap="1.5"
      p="2"
      rounded="md"
    >
      <HStack gap="2" justify="space-between">
        <Text flex="1" fontSize="2xs" fontWeight="600" minW="0" truncate title={sourceLabel}>
          {displayName}
        </Text>
        <Badge colorPalette={badge.palette} flexShrink={0} fontSize="2xs" size="sm" variant="surface">
          {badge.label}
        </Badge>
        <HStack flexShrink={0} gap="0.5">
          {job.status === 'downloading' ? (
            <Tooltip content="Pause download">
              <IconButton
                aria-label="Pause install"
                disabled={isActing}
                size="2xs"
                variant="ghost"
                onClick={() => void runAction(() => pauseModelInstall(job.id), 'Pause failed')}
              >
                <Icon as={PauseIcon} boxSize="3" />
              </IconButton>
            </Tooltip>
          ) : null}
          {job.status === 'paused' ? (
            <Tooltip content="Resume download">
              <IconButton
                aria-label="Resume install"
                disabled={isActing}
                size="2xs"
                variant="ghost"
                onClick={() => void runAction(() => resumeModelInstall(job.id), 'Resume failed')}
              >
                <Icon as={PlayIcon} boxSize="3" />
              </IconButton>
            </Tooltip>
          ) : null}
          {job.status === 'error' ? (
            <Tooltip content="Retry failed download">
              <IconButton
                aria-label="Retry install"
                disabled={isActing}
                size="2xs"
                variant="ghost"
                onClick={() => void runAction(() => restartFailedModelInstall(job.id), 'Retry failed')}
              >
                <Icon as={RotateCcwIcon} boxSize="3" />
              </IconButton>
            </Tooltip>
          ) : null}
          {isActiveInstallStatus(job.status) || job.status === 'paused' ? (
            <Tooltip content="Cancel install">
              <IconButton
                aria-label="Cancel install"
                disabled={isActing}
                size="2xs"
                variant="ghost"
                onClick={() => void runAction(() => cancelModelInstall(job.id), 'Cancel failed')}
              >
                <Icon as={XIcon} boxSize="3" />
              </IconButton>
            </Tooltip>
          ) : null}
        </HStack>
      </HStack>
      {displayName !== sourceLabel ? (
        <Text color="fg.subtle" fontSize="2xs" truncate>
          {sourceLabel}
        </Text>
      ) : null}
      {job.status === 'downloading' || job.status === 'waiting' ? <InstallJobProgress job={job} /> : null}
      {job.status === 'error' && job.error ? (
        <Text color="fg.error" fontSize="2xs" overflowWrap="anywhere">
          {job.error_reason ? `${job.error_reason}: ` : ''}
          {job.error}
        </Text>
      ) : null}
    </Stack>
  );
};

const InstallJobProgress = ({ job }: { job: ModelInstallJob }) => {
  const liveProgress = useInstallProgress(job.id);
  const bytes = liveProgress?.bytes ?? job.bytes ?? 0;
  const totalBytes = liveProgress?.totalBytes ?? job.total_bytes ?? 0;
  const hasTotal = totalBytes > 0;
  const ratio = hasTotal ? Math.min(1, bytes / totalBytes) : null;

  return (
    <Box>
      <Progress.Root aria-label="Download progress" max={1} size="xs" value={ratio}>
        <Progress.Track>
          <Progress.Range />
        </Progress.Track>
      </Progress.Root>
      <Text color="fg.subtle" fontSize="2xs" mt="1">
        {hasTotal
          ? `${Math.round((ratio ?? 0) * 100)}% · ${formatBytes(bytes)} / ${formatBytes(totalBytes)}`
          : 'Waiting for download to start…'}
      </Text>
    </Box>
  );
};
