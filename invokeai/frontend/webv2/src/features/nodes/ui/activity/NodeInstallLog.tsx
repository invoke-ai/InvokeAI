/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { SystemStyleObject } from '@chakra-ui/react';
import type { CustomNodeInstallLogEntry } from '@features/nodes/data/installLogStore';

import { Badge, Icon, Table, Text } from '@chakra-ui/react';
import { useCustomNodeInstallLog } from '@features/nodes/data/installLogStore';
import { Scrollable } from '@platform/ui';
import { EmptyState } from '@platform/ui/EmptyState';
import { InboxIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const INSTALL_LOG_TABLE_SX: SystemStyleObject = {
  '& tbody tr:nth-of-type(odd)': { bg: 'bg.muted' },
  '& th': { bg: 'bg.subtle', position: 'sticky', top: 0, zIndex: 1 },
};

const STATUS_BADGES: Record<CustomNodeInstallLogEntry['status'], { labelKey: string; palette: string }> = {
  completed: { labelKey: 'common.status.completed', palette: 'green' },
  error: { labelKey: 'common.error', palette: 'red' },
  installing: { labelKey: 'nodes.installing', palette: 'blue' },
  uninstalled: { labelKey: 'nodes.uninstalled', palette: 'orange' },
};

/**
 * Session install activity for the nodes manager. Tracks install / uninstall
 * outcomes for the current session in a sticky-header table.
 */
export const NodeInstallLog = () => {
  const { t } = useTranslation();
  const log = useCustomNodeInstallLog();

  if (log.length === 0) {
    return (
      <EmptyState
        description={t('nodes.noRecentActivityDescription')}
        icon={<Icon as={InboxIcon} />}
        title={t('nodes.noRecentActivity')}
      />
    );
  }

  return (
    <Scrollable h="full" label={t('nodes.installActivity')} minH="0">
      <Table.Root css={INSTALL_LOG_TABLE_SX} minW="36rem" size="sm">
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeader borderColor="border.subtle" ps="3">
              {t('common.name')}
            </Table.ColumnHeader>
            <Table.ColumnHeader borderColor="border.subtle">{t('nodes.status')}</Table.ColumnHeader>
            <Table.ColumnHeader borderColor="border.subtle" pe="3">
              {t('nodes.message')}
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
  const { t } = useTranslation();
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
          {t(badge.labelKey)}
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
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
