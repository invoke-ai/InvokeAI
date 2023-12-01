import { Divider, Flex, Heading, Spacer } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import {
  IAINoContentFallback,
  IAINoContentFallbackWithSpinner,
} from 'common/components/IAIImageFallback';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import WorkflowLibraryListItem from 'features/workflowLibrary/components/WorkflowLibraryListItem';
import WorkflowLibraryPagination from 'features/workflowLibrary/components/WorkflowLibraryPagination';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListSystemWorkflowsQuery } from 'services/api/endpoints/workflows';
import { SQLiteDirection, WorkflowRecordOrderBy } from 'services/api/types';

const ORDER_BY_DATA: SelectItem[] = [
  { value: 'created_at', label: 'Created' },
  { value: 'updated_at', label: 'Updated' },
  { value: 'name', label: 'Name' },
];

const DIRECTION_DATA: SelectItem[] = [
  { value: 'ASC', label: 'Ascending' },
  { value: 'DESC', label: 'Descending' },
];

const WorkflowLibraryList = () => {
  const { t } = useTranslation();
  const [page, setPage] = useState(0);
  const [order_by, setOrderBy] = useState<WorkflowRecordOrderBy>('created_at');
  const [direction, setDirection] = useState<SQLiteDirection>('ASC');
  const { data, isLoading, isError, isFetching } =
    useListSystemWorkflowsQuery();

  const handleChangeOrderBy = useCallback(
    (value: string | null) => {
      if (!value || value === order_by) {
        return;
      }
      setOrderBy(value as WorkflowRecordOrderBy);
      setPage(0);
    },
    [order_by]
  );

  const handleChangeDirection = useCallback(
    (value: string | null) => {
      if (!value || value === direction) {
        return;
      }
      setDirection(value as SQLiteDirection);
      setPage(0);
    },
    [direction]
  );

  if (isLoading) {
    return <IAINoContentFallbackWithSpinner label={t('workflows.loading')} />;
  }

  if (!data || isError) {
    return <IAINoContentFallback label={t('workflows.problemLoading')} />;
  }

  if (!data.items.length) {
    return <IAINoContentFallback label={t('workflows.noSystemWorkflows')} />;
  }

  return (
    <>
      <Flex gap={4} alignItems="center" h={10} flexShrink={0} flexGrow={0}>
        <Heading size="md">{t('workflows.systemWorkflows')}</Heading>
        <Spacer />
        <IAIMantineSelect
          label={t('common.orderBy')}
          value={order_by}
          data={ORDER_BY_DATA}
          onChange={handleChangeOrderBy}
          formControlProps={{
            w: '12rem',
            display: 'flex',
            alignItems: 'center',
            gap: 2,
          }}
          disabled={isFetching}
        />
        <IAIMantineSelect
          label={t('common.direction')}
          value={direction}
          data={DIRECTION_DATA}
          onChange={handleChangeDirection}
          formControlProps={{
            w: '12rem',
            display: 'flex',
            alignItems: 'center',
            gap: 2,
          }}
          disabled={isFetching}
        />
      </Flex>
      <Divider />
      <ScrollableContent>
        <Flex w="full" h="full" gap={2} flexDir="column">
          {data.items.map((w) => (
            <WorkflowLibraryListItem key={w.workflow_id} workflowDTO={w} />
          ))}
        </Flex>
      </ScrollableContent>
      <Divider />
      <Flex w="full" justifyContent="space-around">
        <WorkflowLibraryPagination data={data} page={page} setPage={setPage} />
      </Flex>
    </>
  );
};

export default memo(WorkflowLibraryList);
