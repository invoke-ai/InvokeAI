import type { SystemStyleObject } from '@chakra-ui/react';
import type { CustomNodeInstallLogEntry } from '@workbench/customNodes/installLogStore';

import { Badge, Icon, Table, Text } from '@chakra-ui/react';
import { Scrollable } from '@workbench/components/ui';
import { EmptyState } from '@workbench/components/ui/EmptyState';
import { useCustomNodeInstallLog } from '@workbench/customNodes/installLogStore';
import { InboxIcon } from 'lucide-react';

const INSTALL_LOG_TABLE_SX: SystemStyleObject = {
  '& tbody tr:nth-of-type(odd)': { bg: 'bg.muted' },
  '& th': { bg: 'bg.subtle', position: 'sticky', top: 0, zIndex: 1 },
};

const STATUS_BADGES: Record<CustomNodeInstallLogEntry['status'], { label: string; palette: string }> = {
  completed: { label: 'Completed', palette: 'green' },
  error: { label: 'Error', palette: 'red' },
  installing: { label: 'Installing', palette: 'blue' },
  uninstalled: { label: 'Uninstalled', palette: 'orange' },
};

/**
 * Session install activity for the nodes manager. Tracks install / uninstall
 * outcomes for the current session in a sticky-header table.
 */
export const NodeInstallLog = () => {
  const log = useCustomNodeInstallLog();

  if (log.length === 0) {
    return (
      <EmptyState
        description="Install or uninstall a node pack and the outcome shows up here."
        icon={<Icon as={InboxIcon} />}
        title="No recent activity"
      />
    );
  }

  return (
    <Scrollable h="full" label="Custom node install activity" minH="0">
      <Table.Root css={INSTALL_LOG_TABLE_SX} minW="36rem" size="sm">
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeader borderColor="border.subtle" ps="3">
              Name
            </Table.ColumnHeader>
            <Table.ColumnHeader borderColor="border.subtle">Status</Table.ColumnHeader>
            <Table.ColumnHeader borderColor="border.subtle" pe="3">
              Message
            </Table.ColumnHeader>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {log.map((entry) => (
            <InstallLogRow key={entry.id} entry={entry} />
          ))}
        </Table.Body>
      </Table.Root>
    </Scrollable>
  );
};

const InstallLogRow = ({ entry }: { entry: CustomNodeInstallLogEntry }) => {
  const badge = STATUS_BADGES[entry.status];

  return (
    <Table.Row>
      <Table.Cell borderColor="border.subtle" ps="3">
        <Text fontSize="xs" title={entry.name} truncate>
          {entry.name}
        </Text>
      </Table.Cell>
      <Table.Cell borderColor="border.subtle">
        <Badge colorPalette={badge.palette} fontSize="2xs" variant="surface">
          {badge.label}
        </Badge>
      </Table.Cell>
      <Table.Cell borderColor="border.subtle" pe="3">
        <Text color="fg.subtle" fontSize="2xs" lineClamp={2} overflowWrap="anywhere">
          {entry.message ?? ''}
        </Text>
      </Table.Cell>
    </Table.Row>
  );
};
