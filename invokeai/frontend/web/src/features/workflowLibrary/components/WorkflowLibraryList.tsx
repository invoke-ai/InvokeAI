import { CloseIcon } from '@chakra-ui/icons';
import {
  Divider,
  Flex,
  InputGroup,
  InputRightElement,
  Spacer,
} from '@chakra-ui/react';
import {
  IAINoContentFallback,
  IAINoContentFallbackWithSpinner,
} from 'common/components/IAIImageFallback';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvButtonGroup } from 'common/components/InvButtonGroup/InvButtonGroup';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvInput } from 'common/components/InvInput/InvInput';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import WorkflowLibraryListItem from 'features/workflowLibrary/components/WorkflowLibraryListItem';
import WorkflowLibraryPagination from 'features/workflowLibrary/components/WorkflowLibraryPagination';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import type {
  SQLiteDirection,
  WorkflowRecordOrderBy,
} from 'services/api/types';
import { useDebounce } from 'use-debounce';
import { z } from 'zod';

const PER_PAGE = 10;

const zOrderBy = z.enum(['opened_at', 'created_at', 'updated_at', 'name']);
type OrderBy = z.infer<typeof zOrderBy>;
const isOrderBy = (v: unknown): v is OrderBy => zOrderBy.safeParse(v).success;
const ORDER_BY_OPTIONS: InvSelectOption[] = [
  { value: 'opened_at', label: 'Opened' },
  { value: 'created_at', label: 'Created' },
  { value: 'updated_at', label: 'Updated' },
  { value: 'name', label: 'Name' },
];

const zDirection = z.enum(['ASC', 'DESC']);
type Direction = z.infer<typeof zDirection>;
const isDirection = (v: unknown): v is Direction =>
  zDirection.safeParse(v).success;
const DIRECTION_OPTIONS: InvSelectOption[] = [
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

  const queryArg = useMemo<Parameters<typeof useListWorkflowsQuery>[0]>(() => {
    if (category === 'user') {
      return {
        page,
        per_page: PER_PAGE,
        order_by,
        direction,
        category,
        query: debouncedQuery,
      };
    }
    return {
      page,
      per_page: PER_PAGE,
      order_by: 'name' as const,
      direction: 'ASC' as const,
      category,
      query: debouncedQuery,
    };
  }, [category, debouncedQuery, direction, order_by, page]);

  const { data, isLoading, isError, isFetching } =
    useListWorkflowsQuery(queryArg);

  const onChangeOrderBy = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isOrderBy(v?.value) || v.value === order_by) {
        return;
      }
      setOrderBy(v.value);
      setPage(0);
    },
    [order_by]
  );
  const valueOrderBy = useMemo(
    () => ORDER_BY_OPTIONS.find((o) => o.value === order_by),
    [order_by]
  );

  const onChangeDirection = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isDirection(v?.value) || v.value === direction) {
        return;
      }
      setDirection(v.value);
      setPage(0);
    },
    [direction]
  );
  const valueDirection = useMemo(
    () => DIRECTION_OPTIONS.find((o) => o.value === direction),
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

  return (
    <>
      <Flex gap={4} alignItems="center" h={10} flexShrink={0} flexGrow={0}>
        <InvButtonGroup>
          <InvButton
            variant={category === 'user' ? undefined : 'ghost'}
            onClick={handleSetUserCategory}
            isChecked={category === 'user'}
          >
            {t('workflows.userWorkflows')}
          </InvButton>
          <InvButton
            variant={category === 'default' ? undefined : 'ghost'}
            onClick={handleSetDefaultCategory}
            isChecked={category === 'default'}
          >
            {t('workflows.defaultWorkflows')}
          </InvButton>
        </InvButtonGroup>
        <Spacer />
        {category === 'user' && (
          <>
            <InvControl
              label={t('common.orderBy')}
              isDisabled={isFetching}
              w={64}
              minW={56}
            >
              <InvSelect
                value={valueOrderBy}
                options={ORDER_BY_OPTIONS}
                onChange={onChangeOrderBy}
              />
            </InvControl>
            <InvControl
              label={t('common.direction')}
              isDisabled={isFetching}
              w={64}
              minW={56}
            >
              <InvSelect
                value={valueDirection}
                options={DIRECTION_OPTIONS}
                onChange={onChangeDirection}
              />
            </InvControl>
          </>
        )}
        <InputGroup w="20rem">
          <InvInput
            placeholder={t('workflows.searchWorkflows')}
            value={query}
            onKeyDown={handleKeydownFilterText}
            onChange={handleChangeFilterText}
            data-testid="workflow-search-input"
            minW={64}
          />
          {query.trim().length && (
            <InputRightElement>
              <InvIconButton
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
          <Flex w="full" h="full" flexDir="column">
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
