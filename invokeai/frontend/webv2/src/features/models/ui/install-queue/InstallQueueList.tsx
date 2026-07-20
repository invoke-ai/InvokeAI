/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelInstallJob } from '@features/models/core/types';

import { Flex, Spinner, Stack, Text } from '@chakra-ui/react';
import { refreshInstalls } from '@features/models/data/installsStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { Button, Scrollable } from '@platform/ui';
import { useTranslation } from 'react-i18next';

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
  const { t } = useTranslation();
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
          {t('models.couldNotLoadInstallQueue')}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {error}
        </Text>
        <Button mt="1" size="xs" variant="outline" onClick={() => void refreshInstalls()}>
          {t('common.retry')}
        </Button>
      </Stack>
    );
  }

  if (jobs.length === 0) {
    return (
      <Stack align="center" gap="1" py="6">
        <Text color="fg.muted" fontSize="xs" fontWeight="600">
          {t('models.noInstallsYet')}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {t('models.noInstallsDescription')}
        </Text>
      </Stack>
    );
  }

  return (
    <Scrollable h="full" label={t('models.installJobs')} minH="0">
      <Stack gap="1.5">
        {jobs.map((job) => (
          <InstallJobRow key={job.id} job={job} onError={notify.error} />
        ))}
      </Stack>
    </Scrollable>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
