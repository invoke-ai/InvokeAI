import type { ModelInstallJob } from '@workbench/models/types';

import { Badge, Box, Collapsible, Flex, HStack, Icon, Spinner, Text } from '@chakra-ui/react';
import { Button, IconButton, Tooltip } from '@workbench/components/ui';
import {
  cancelModelInstall,
  pauseModelInstall,
  pruneCompletedModelInstalls,
  resumeModelInstall,
} from '@workbench/models/api';
import {
  ensureInstallsLoaded,
  isActiveInstallStatus,
  refreshInstalls,
  useInstallsSnapshot,
} from '@workbench/models/installsStore';
import { setQueueExpanded, useModelsUi } from '@workbench/models/uiStore';
import { useNotify } from '@workbench/useNotify';
import { ChevronUpIcon, ListOrderedIcon, PauseIcon, PlayIcon, RefreshCcwIcon, Trash2Icon, XIcon } from 'lucide-react';
import { useEffect, useState } from 'react';

import { InstallQueueList } from './InstallQueueList';
import { getInstallJobDisplayName } from './queueUtils';

/** Persistent, collapsible install queue footer for the model manager detail pane. */
export const InstallQueueBar = () => {
  const notify = useNotify();
  const { error, jobs, status } = useInstallsSnapshot();
  const { queueExpanded } = useModelsUi();

  useEffect(() => {
    ensureInstallsLoaded();
  }, []);

  const activeJobs = jobs.filter((job) => isActiveInstallStatus(job.status));
  const pausedJobs = jobs.filter((job) => job.status === 'paused');
  const pausableJobs = jobs.filter((job) => job.status === 'downloading');
  const cancellableJobs = jobs.filter((job) => isActiveInstallStatus(job.status) || job.status === 'paused');
  const settledCount = jobs.filter((job) => !isActiveInstallStatus(job.status) && job.status !== 'paused').length;

  const [busyAction, setBusyAction] = useState<'pause' | 'resume' | 'cancel' | 'prune' | 'refresh' | null>(null);

  const runBulk = async (action: 'pause' | 'resume' | 'cancel', targets: ModelInstallJob[]) => {
    const call = action === 'pause' ? pauseModelInstall : action === 'resume' ? resumeModelInstall : cancelModelInstall;

    setBusyAction(action);

    try {
      await Promise.all(targets.map((job) => call(job.id)));
      await refreshInstalls();
    } catch (bulkError) {
      notify.error('Queue action failed', bulkError instanceof Error ? bulkError.message : String(bulkError));
      void refreshInstalls();
    } finally {
      setBusyAction(null);
    }
  };

  const handlePrune = async () => {
    setBusyAction('prune');

    try {
      await pruneCompletedModelInstalls();
      await refreshInstalls();
    } catch (pruneError) {
      notify.error('Prune failed', pruneError instanceof Error ? pruneError.message : String(pruneError));
    } finally {
      setBusyAction(null);
    }
  };

  const handleRefresh = async () => {
    setBusyAction('refresh');

    try {
      await refreshInstalls();
    } finally {
      setBusyAction(null);
    }
  };

  const summary =
    activeJobs.length > 0
      ? `${getInstallJobDisplayName(activeJobs[0]!)}${activeJobs.length > 1 ? ` +${activeJobs.length - 1} more` : ''}`
      : jobs.length > 0
        ? `${jobs.length} job${jobs.length === 1 ? '' : 's'} · no active installs`
        : 'No installs yet';

  return (
    <Collapsible.Root
      bg="bg.subtle"
      borderTopWidth={1}
      flexShrink={0}
      open={queueExpanded}
      overflow="hidden"
      onOpenChange={(event) => setQueueExpanded(event.open)}
    >
      <Collapsible.Content>
        <Flex direction="column" h="min(22rem, 45dvh)" minH="0" overflow="hidden">
          <HStack borderBottomWidth={1} gap="2" justify="space-between" px="3" py="1.5">
            <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
              Install Queue
            </Text>
            <HStack gap="1">
              <Button
                loading={busyAction === 'refresh'}
                size="2xs"
                variant="ghost"
                onClick={() => void handleRefresh()}
              >
                <Icon as={RefreshCcwIcon} boxSize="3" />
                Refresh
              </Button>
              <Button
                disabled={settledCount === 0}
                loading={busyAction === 'prune'}
                size="2xs"
                variant="ghost"
                onClick={() => void handlePrune()}
              >
                <Icon as={Trash2Icon} boxSize="3" />
                Clear finished
              </Button>
            </HStack>
          </HStack>
          <Box flex="1" minH="0" overflow="hidden" p="2">
            <InstallQueueList error={error} jobs={jobs} status={status} />
          </Box>
        </Flex>
      </Collapsible.Content>

      <HStack gap="1" px="3" py="2">
        <Collapsible.Trigger
          alignItems="center"
          bg="transparent"
          color="inherit"
          display="flex"
          flex="1"
          gap="2"
          minW="0"
          textAlign="start"
          _hover={{ color: 'fg' }}
        >
          {activeJobs.length > 0 ? (
            <Spinner borderWidth="1.5px" boxSize="3.5" color="accent.solid" flexShrink={0} />
          ) : (
            <Icon as={ListOrderedIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
          )}
          <Text flex="1" fontSize="xs" fontWeight="600" minW="0" truncate>
            {summary}
          </Text>

          {activeJobs.length > 0 ? (
            <Badge colorPalette="accent" flexShrink={0} fontSize="2xs" size="sm" variant="solid">
              {activeJobs.length}
            </Badge>
          ) : null}
          <Collapsible.Indicator
            _open={{ transform: 'rotate(180deg)' }}
            transition="transform var(--wb-motion-duration-slow)"
          >
            <Icon as={ChevronUpIcon} boxSize="4" color="fg.subtle" flexShrink={0} />
          </Collapsible.Indicator>
        </Collapsible.Trigger>

        <HStack flexShrink={0} gap="0.5">
          {pausableJobs.length > 0 ? (
            <Tooltip content="Pause all downloads">
              <IconButton
                aria-label="Pause all installs"
                loading={busyAction === 'pause'}
                size="2xs"
                variant="ghost"
                onClick={() => void runBulk('pause', pausableJobs)}
              >
                <Icon as={PauseIcon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
          ) : null}
          {pausedJobs.length > 0 ? (
            <Tooltip content="Resume all paused downloads">
              <IconButton
                aria-label="Resume all installs"
                loading={busyAction === 'resume'}
                size="2xs"
                variant="ghost"
                onClick={() => void runBulk('resume', pausedJobs)}
              >
                <Icon as={PlayIcon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
          ) : null}
          {cancellableJobs.length > 0 ? (
            <Tooltip content="Cancel all installs">
              <IconButton
                aria-label="Cancel all installs"
                loading={busyAction === 'cancel'}
                size="2xs"
                variant="ghost"
                onClick={() => void runBulk('cancel', cancellableJobs)}
              >
                <Icon as={XIcon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
          ) : null}
        </HStack>
      </HStack>
    </Collapsible.Root>
  );
};
