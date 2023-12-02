import { CloseIcon } from '@chakra-ui/icons';
import {
  ButtonGroup,
  Divider,
  Flex,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Spacer,
} from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import IAIButton from 'common/components/IAIButton';
import {
  IAINoContentFallback,
  IAINoContentFallbackWithSpinner,
} from 'common/components/IAIImageFallback';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import { WorkflowCategory } from 'features/nodes/types/workflow';
import WorkflowLibraryListItem from 'features/workflowLibrary/components/WorkflowLibraryListItem';
import WorkflowLibraryPagination from 'features/workflowLibrary/components/WorkflowLibraryPagination';
import {
  ChangeEvent,
  KeyboardEvent,
  memo,
  useCallback,
  useMemo,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import { SQLiteDirection, WorkflowRecordOrderBy } from 'services/api/types';
import { useDebounce } from 'use-debounce';

const PER_PAGE = 10;

const ORDER_BY_DATA: SelectItem[] = [
  { value: 'opened_at', label: 'Opened' },
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
  const [category, setCategory] = useState<WorkflowCategory>('user');
  const [page, setPage] = useState(0);
  const [filter_text, setFilterText] = useState('');
  const [order_by, setOrderBy] = useState<WorkflowRecordOrderBy>('opened_at');
  const [direction, setDirection] = useState<SQLiteDirection>('ASC');
  const [debouncedFilterText] = useDebounce(filter_text, 500);

  const query = useMemo(() => {
    if (category === 'user') {
      return {
        page,
        per_page: PER_PAGE,
        order_by,
        direction,
        category,
        filter_text: debouncedFilterText,
      };
    }
    return {
      page,
      per_page: PER_PAGE,
      order_by: 'name' as const,
      direction: 'ASC' as const,
      category,
      filter_text: debouncedFilterText,
    };
  }, [category, debouncedFilterText, direction, order_by, page]);

  const { data, isLoading, isError, isFetching } = useListWorkflowsQuery(query);

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

  const handleSetUserCategory = useCallback(() => {
    setCategory('user');
    setPage(0);
  }, []);

  const handleSetDefaultCategory = useCallback(() => {
    setCategory('default');
    setPage(0);
  }, []);

  return (
    <>
      <Flex gap={4} alignItems="center" h={10} flexShrink={0} flexGrow={0}>
        <ButtonGroup>
          <IAIButton
            variant={category === 'user' ? undefined : 'ghost'}
            onClick={handleSetUserCategory}
            isChecked={category === 'user'}
          >
            {t('workflows.userWorkflows')}
          </IAIButton>
          <IAIButton
            variant={category === 'default' ? undefined : 'ghost'}
            onClick={handleSetDefaultCategory}
            isChecked={category === 'default'}
          >
            {t('workflows.defaultWorkflows')}
          </IAIButton>
        </ButtonGroup>
        <Spacer />
        {category === 'user' && (
          <>
            <IAIMantineSelect
              label={t('common.orderBy')}
              value={order_by}
              data={ORDER_BY_DATA}
              onChange={handleChangeOrderBy}
              formControlProps={{
                w: 48,
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
                w: 48,
                display: 'flex',
                alignItems: 'center',
                gap: 2,
              }}
              disabled={isFetching}
            />
          </>
        )}
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
      </Flex>
      <Divider />
      {isLoading ? (
        <IAINoContentFallbackWithSpinner label={t('workflows.loading')} />
      ) : !data || isError ? (
        <IAINoContentFallback label={t('workflows.problemLoading')} />
      ) : data.items.length ? (
        <ScrollableContent>
          <Flex w="full" h="full" gap={2} px={1} flexDir="column">
            {data.items.map((w) => (
              <WorkflowLibraryListItem key={w.workflow_id} workflowDTO={w} />
            ))}
          </Flex>
        </ScrollableContent>
      ) : (
        <IAINoContentFallback label={t('workflows.noUserWorkflows')} />
      )}
      <Divider />
      {data && (
        <Flex w="full" justifyContent="space-around">
          <WorkflowLibraryPagination
            data={data}
            page={page}
            setPage={setPage}
          />
        </Flex>
      )}
    </>
  );
};

export default memo(WorkflowLibraryList);
