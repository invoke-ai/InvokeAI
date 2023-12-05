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
import { useStore } from '@nanostores/react';
import { $projectId } from 'app/store/nanostores/projectId';
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

const BASE_ORDER_BY_DATA: SelectItem[] = [
  { value: 'created_at', label: 'Created' },
  { value: 'updated_at', label: 'Updated' },
  { value: 'name', label: 'Name' },
];

const OPENED_AT_ORDER_BY_EXTRAS: SelectItem[] = [
  { value: 'opened_at', label: 'Opened' },
];

const DIRECTION_DATA: SelectItem[] = [
  { value: 'ASC', label: 'Ascending' },
  { value: 'DESC', label: 'Descending' },
];

const WorkflowLibraryList = () => {
  const { t } = useTranslation();
  const [category, setCategory] = useState<WorkflowCategory>('user');
  const [page, setPage] = useState(0);
  const [query, setQuery] = useState('');
  const [order_by, setOrderBy] = useState<WorkflowRecordOrderBy>('opened_at');
  const [direction, setDirection] = useState<SQLiteDirection>('ASC');
  const [debouncedQuery] = useDebounce(query, 500);
  const projectId = useStore($projectId);
  const orderByData = useMemo<SelectItem[]>(() => {
    if (category === 'project') {
      return BASE_ORDER_BY_DATA;
    }
    return [...BASE_ORDER_BY_DATA, ...OPENED_AT_ORDER_BY_EXTRAS];
  }, [category]);

  const queryArg = useMemo<Parameters<typeof useListWorkflowsQuery>[0]>(() => {
    if (category === 'default') {
      return {
        page,
        per_page: PER_PAGE,
        order_by: 'name' as const,
        direction: 'ASC' as const,
        category,
        query: debouncedQuery,
      };
    }
    return {
      page,
      per_page: PER_PAGE,
      order_by,
      direction,
      category,
      query: debouncedQuery,
    };
  }, [category, debouncedQuery, direction, order_by, page]);

  const { data, isLoading, isError, isFetching } =
    useListWorkflowsQuery(queryArg);

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
    setQuery('');
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
      setQuery(e.target.value);
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

  const handleSetProjectCategory = useCallback(() => {
    setCategory('project');
    setPage(0);
    // Projects can't be sorted by opened_at
    if (order_by === 'opened_at') {
      setOrderBy('name');
    }
  }, [order_by]);

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
          {projectId && (
            <IAIButton
              variant={category === 'project' ? undefined : 'ghost'}
              onClick={handleSetProjectCategory}
              isChecked={category === 'project'}
            >
              {t('workflows.projectWorkflows')}
            </IAIButton>
          )}
          <IAIButton
            variant={category === 'default' ? undefined : 'ghost'}
            onClick={handleSetDefaultCategory}
            isChecked={category === 'default'}
          >
            {t('workflows.defaultWorkflows')}
          </IAIButton>
        </ButtonGroup>
        <Spacer />
        {category !== 'default' && (
          <>
            <IAIMantineSelect
              label={t('common.orderBy')}
              value={order_by}
              data={orderByData}
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
            value={query}
            onKeyDown={handleKeydownFilterText}
            onChange={handleChangeFilterText}
            data-testid="workflow-search-input"
          />
          {query.trim().length && (
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
        <IAINoContentFallback label={t('workflows.noWorkflows')} />
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
