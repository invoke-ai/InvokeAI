import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Box, Button, Flex, Heading, Table, Tbody, Td, Text, Th, Thead, Tr } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBroomBold } from 'react-icons/pi';

import type { InstallLogEntry } from './useCustomNodesInstallLog';
import { useCustomNodesInstallLog } from './useCustomNodesInstallLog';

const tableSx: SystemStyleObject = {
  '& tbody tr:nth-of-type(odd)': {
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
  },
  '& tbody tr:nth-of-type(even)': {
    backgroundColor: 'transparent',
  },
  'td, th': {
    borderColor: 'base.700',
  },
  th: {
    position: 'sticky',
    top: 0,
    zIndex: 1,
    backgroundColor: 'base.800',
    py: 2,
  },
  'th:first-of-type': {
    borderTopLeftRadius: 'base',
  },
  'th:last-of-type': {
    borderTopRightRadius: 'base',
  },
  'tr:last-of-type td:first-of-type': {
    borderBottomLeftRadius: 'base',
  },
  'tr:last-of-type td:last-of-type': {
    borderBottomRightRadius: 'base',
  },
};

const getStatusColor = (status: InstallLogEntry['status']) => {
  switch (status) {
    case 'installing':
      return 'invokeBlue';
    case 'completed':
      return 'invokeGreen';
    case 'error':
      return 'error';
    case 'uninstalled':
      return 'invokeYellow';
    default:
      return 'base';
  }
};

export const CustomNodesInstallLog = memo(() => {
  const { t } = useTranslation();
  const { log, clearLog } = useCustomNodesInstallLog();

  return (
    <Flex flexDir="column" h="full" gap={4}>
      <Flex justifyContent="space-between" alignItems="center">
        <Heading size="md">{t('customNodes.installQueue')}</Heading>
        <Button leftIcon={<PiBroomBold />} size="sm" isDisabled={log.length === 0} onClick={clearLog} variant="outline">
          {t('customNodes.clearLog')}
        </Button>
      </Flex>

      <Box layerStyle="second" borderRadius="base" w="full" h="full">
        <ScrollableContent>
          <Table size="sm" sx={tableSx}>
            <Thead>
              <Tr>
                <Th width="50%">{t('customNodes.name')}</Th>
                <Th width="25%">{t('customNodes.status')}</Th>
                <Th width="25%">{t('customNodes.message')}</Th>
              </Tr>
            </Thead>
            <Tbody>
              {log.length === 0 ? (
                <Tr>
                  <Td colSpan={3} textAlign="center" py={8}>
                    <Text variant="subtext">{t('customNodes.queueEmpty')}</Text>
                  </Td>
                </Tr>
              ) : (
                log.map((entry) => (
                  <Tr key={entry.id}>
                    <Td>
                      <Text fontSize="sm" noOfLines={1}>
                        {entry.name}
                      </Text>
                    </Td>
                    <Td>
                      <Badge colorScheme={getStatusColor(entry.status)}>{entry.status}</Badge>
                    </Td>
                    <Td>
                      <Text fontSize="xs" color="base.400" noOfLines={2}>
                        {entry.message}
                      </Text>
                    </Td>
                  </Tr>
                ))
              )}
            </Tbody>
          </Table>
        </ScrollableContent>
      </Box>
    </Flex>
  );
});

CustomNodesInstallLog.displayName = 'CustomNodesInstallLog';
