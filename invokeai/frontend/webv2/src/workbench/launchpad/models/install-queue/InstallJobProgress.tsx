import type { ModelInstallJob } from '@workbench/models/types';

import { Box, Progress, Text } from '@chakra-ui/react';
import { useInstallProgress } from '@workbench/models/installsStore';
import { formatBytes } from '@workbench/models/taxonomy';

export const InstallJobProgress = ({ job }: { job: ModelInstallJob }) => {
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
