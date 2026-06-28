import { Flex, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useGetSystemStatusQuery } from 'services/api/endpoints/appInfo';

const formatPercent = (value: number | null): string => (value === null ? '--%' : `${Math.round(value)}%`);

const SystemHardwareStatus = () => {
  const { data } = useGetSystemStatusQuery(undefined, { pollingInterval: 3000 });

  if (!data) {
    return null;
  }

  return (
    <Flex w="full" flexDir="column" gap={1} pt={2} borderTopWidth={1} borderColor="base.700">
      <Text fontSize="xs" color="base.300">
        CPU: {Math.round(data.cpu_percent)}% {data.cpu_frequency_ghz?.toFixed(2) ?? '--'}GHz
      </Text>
      <Text fontSize="xs" color="base.300">
        MEMORY: {data.memory_used_gb.toFixed(1)}/{data.memory_total_gb.toFixed(1)}GB ({Math.round(data.memory_percent)}
        %)
      </Text>
      {data.gpus.map((gpu) => (
        <Text key={gpu.index} fontSize="xs" color="base.300">
          GPU{gpu.index}: {formatPercent(gpu.utilization_percent)} utilization - loaded {gpu.loaded_gb.toFixed(1)}
          GB, total {gpu.total_gb.toFixed(1)}GB
        </Text>
      ))}
    </Flex>
  );
};

export default memo(SystemHardwareStatus);
