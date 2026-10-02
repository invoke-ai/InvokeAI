import type { SystemStyleObject } from '@chakra-ui/react';

import { Flex, Icon, Spinner, Table, VisuallyHidden } from '@chakra-ui/react';
import { refreshInstalls } from '@features/models/data/installsStore';
import { Button, Scrollable } from '@platform/ui';
import { EmptyState } from '@platform/ui/EmptyState';
import { DownloadIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { InstallQueueRow } from './queueModel';

import { InstallJobRow } from './InstallJobRow';

const TABLE_SX: SystemStyleObject = {
  '& th': { bg: 'bg.subtle', position: 'sticky', top: 0, zIndex: 1 },
  '& tr[data-tone="error"] > td': { bg: 'bg.error' },
  '& tr[data-tone="error"] > td:first-of-type': { boxShadow: 'inset 2px 0 0 {colors.border.error}' },
  '& tr[data-tone="warning"] > td': { bg: 'bg.warning' },
  '& tr[data-tone="warning"] > td:first-of-type': { boxShadow: 'inset 2px 0 0 {colors.fg.warning}' },
};

const HEADER_PROPS = {
  borderColor: 'border.subtle',
  color: 'fg.muted',
  fontSize: '2xs',
  fontWeight: '700',
  letterSpacing: 'wider',
  py: '1.5',
  textTransform: 'uppercase',
} as const;

export const InstallQueueTable = ({
  error,
  rows,
  status,
}: {
  error: string | null;
  rows: InstallQueueRow[];
  status: 'idle' | 'loading' | 'loaded' | 'error';
}) => {
  const { t } = useTranslation();

  if (status === 'loading' || status === 'idle') {
    return (
      <Flex align="center" h="full" justify="center" py="6">
        <Spinner color="fg.muted" size="sm" />
      </Flex>
    );
  }

  if (status === 'error') {
    return (
      <EmptyState danger description={error ?? undefined} h="full" title={t('models.couldNotLoadInstallQueue')}>
        <Button size="xs" variant="outline" onClick={() => void refreshInstalls()}>
          {t('common.retry')}
        </Button>
      </EmptyState>
    );
  }

  if (rows.length === 0) {
    return (
      <EmptyState
        description={t('models.noInstallsDescription')}
        h="full"
        icon={<Icon as={DownloadIcon} />}
        title={t('models.noInstallsYet')}
      />
    );
  }

  return (
    <Scrollable h="full" label={t('models.installJobs')} minH="0">
      <Table.Root css={TABLE_SX} minW="32rem" size="sm" tableLayout="fixed" w="full">
        <Table.ColumnGroup>
          <Table.Column w="2.25rem" />
          <Table.Column />
          <Table.Column w="30%" />
          <Table.Column w="7.75rem" />
          <Table.Column w="4.5rem" />
        </Table.ColumnGroup>
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeader {...HEADER_PROPS} ps="3">
              <VisuallyHidden>{t('models.queueColumns.state')}</VisuallyHidden>
            </Table.ColumnHeader>
            <Table.ColumnHeader {...HEADER_PROPS}>{t('models.queueColumns.name')}</Table.ColumnHeader>
            <Table.ColumnHeader {...HEADER_PROPS}>{t('models.queueColumns.progress')}</Table.ColumnHeader>
            <Table.ColumnHeader {...HEADER_PROPS}>{t('models.queueColumns.status')}</Table.ColumnHeader>
            <Table.ColumnHeader {...HEADER_PROPS} pe="2" textAlign="end">
              {t('models.queueColumns.actions')}
            </Table.ColumnHeader>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {rows.map((row) => (
            <InstallJobRow key={row.job.id} row={row} />
          ))}
        </Table.Body>
      </Table.Root>
    </Scrollable>
  );
};
