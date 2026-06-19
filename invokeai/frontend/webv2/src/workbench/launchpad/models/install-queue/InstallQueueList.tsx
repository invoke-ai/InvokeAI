import type { ModelInstallJob } from '@workbench/models/types';

import { Flex, Spinner, Stack, Text } from '@chakra-ui/react';
import { Button, Scrollable } from '@workbench/components/ui';
import { refreshInstalls } from '@workbench/models/installsStore';
import { useNotify } from '@workbench/useNotify';

import { InstallJobRow } from './InstallJobRow';

export const InstallQueueList = ({
  error,
  jobs,
  status,
}: {
  error: string | null;
  jobs: ModelInstallJob[];
  status: 'idle' | 'loading' | 'loaded' | 'error';
}) => {
  const notify = useNotify();

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
    <Scrollable h="full" label="Install jobs" minH="0">
      <Stack gap="1.5">
        {jobs.map((job) => (
          <InstallJobRow key={job.id} job={job} onError={notify.error} />
        ))}
      </Stack>
    </Scrollable>
  );
};
