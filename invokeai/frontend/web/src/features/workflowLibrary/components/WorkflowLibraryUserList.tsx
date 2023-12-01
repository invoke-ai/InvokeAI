import { CloseIcon } from '@chakra-ui/icons';
import {
  Divider,
  Flex,
  Heading,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Spacer,
} from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import {
  IAINoContentFallback,
  IAINoContentFallbackWithSpinner,
} from 'common/components/IAIImageFallback';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import WorkflowLibraryListItem from 'features/workflowLibrary/components/WorkflowLibraryListItem';
import WorkflowLibraryPagination from 'features/workflowLibrary/components/WorkflowLibraryPagination';
import { ChangeEvent, KeyboardEvent, memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import { SQLiteDirection, WorkflowRecordOrderBy } from 'services/api/types';
import { useDebounce } from 'use-debounce';

const PER_PAGE = 10;

const ORDER_BY_DATA: SelectItem[] = [
  { value: 'opened_at', label: 'Recently Opened' },
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
  const [filter_text, setFilterText] = useState('');
  const [order_by, setOrderBy] = useState<WorkflowRecordOrderBy>('opened_at');
  const [direction, setDirection] = useState<SQLiteDirection>('ASC');
  const [debouncedFilterText] = useDebounce(filter_text, 500, {
    leading: true,
  });
  const { data, isLoading, isError, isFetching } = useListWorkflowsQuery({
    page,
    per_page: PER_PAGE,
    order_by,
    direction,
    filter_name: debouncedFilterText,
  });

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

  const resetFilterText = useCallback(() => {
    setFilterText('');
    setPage(0);
  }, []);

  const handleKeydownFilterText = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      // exit search mode on escape
      if (e.key === 'Escape') {
        resetFilterText();
        e.preventDefault();
        setPage(0);
      }
    },
    [resetFilterText]
  );

  const handleChangeFilterText = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setFilterText(e.target.value);
      setPage(0);
    },
    []
  );

  if (isLoading) {
    return <IAINoContentFallbackWithSpinner label={t('workflows.loading')} />;
  }

  if (!data || isError) {
    return <IAINoContentFallback label={t('workflows.problemLoading')} />;
  }

  return (
    <>
      <Flex gap={4} alignItems="center" h={10} flexShrink={0} flexGrow={0}>
        <Heading size="md">{t('workflows.userWorkflows')}</Heading>
        <Spacer />
        <InputGroup w="20rem">
          <Input
            placeholder={t('workflows.searchWorkflows')}
            value={filter_text}
            onKeyDown={handleKeydownFilterText}
            onChange={handleChangeFilterText}
            data-testid="workflow-search-input"
          />
          {filter_text.trim().length && (
            <InputRightElement>
              <IconButton
                onClick={resetFilterText}
                size="xs"
                variant="ghost"
                aria-label={t('workflows.clearWorkflowSearchFilter')}
                opacity={0.5}
                icon={<CloseIcon boxSize={2} />}
              />
            </InputRightElement>
          )}
        </InputGroup>
        <IAIMantineSelect
          label={t('common.orderBy')}
          value={order_by}
          data={ORDER_BY_DATA}
          onChange={handleChangeOrderBy}
          formControlProps={{
            w: '15rem',
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
      {data.items.length ? (
        <ScrollableContent>
          <Flex w="full" h="full" gap={2} flexDir="column">
            {data.items.map((w) => (
              <WorkflowLibraryListItem key={w.workflow_id} workflowDTO={w} />
            ))}
          </Flex>
        </ScrollableContent>
      ) : (
        <IAINoContentFallback label={t('workflows.noUserWorkflows')} />
      )}
      <Divider />
      <Flex w="full" justifyContent="space-around">
        <WorkflowLibraryPagination data={data} page={page} setPage={setPage} />
      </Flex>
    </>
  );
};

export default memo(WorkflowLibraryList);
