import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  ButtonGroup,
  Combobox,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $projectId } from 'app/store/nanostores/projectId';
import { $workflowCategories } from 'app/store/nanostores/workflowCategories';
import { IAINoContentFallback, IAINoContentFallbackWithSpinner } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import WorkflowLibraryListItem from 'features/workflowLibrary/components/WorkflowLibraryListItem';
import WorkflowLibraryPagination from 'features/workflowLibrary/components/WorkflowLibraryPagination';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import type { SQLiteDirection, WorkflowRecordOrderBy } from 'services/api/types';
import { useDebounce } from 'use-debounce';
import { z } from 'zod';

import UploadWorkflowButton from './UploadWorkflowButton';

const PER_PAGE = 10;

const zOrderBy = z.enum(['opened_at', 'created_at', 'updated_at', 'name']);
type OrderBy = z.infer<typeof zOrderBy>;
const isOrderBy = (v: unknown): v is OrderBy => zOrderBy.safeParse(v).success;

const zDirection = z.enum(['ASC', 'DESC']);
type Direction = z.infer<typeof zDirection>;
const isDirection = (v: unknown): v is Direction => zDirection.safeParse(v).success;


const WorkflowLibraryList = () => {
  const { t } = useTranslation();
  const workflowCategories = useStore($workflowCategories);
  const [selectedCategory, setSelectedCategory] = useState<WorkflowCategory>('user');
  const [page, setPage] = useState(0);
  const [query, setQuery] = useState('');
  const projectId = useStore($projectId);

  const ORDER_BY_OPTIONS: ComboboxOption[] = useMemo(() => [
    { value: 'opened_at', label: t('workflows.opened') },
    { value: 'created_at', label: t('workflows.created') },
    { value: 'updated_at', label: t('workflows.updated') },
    { value: 'name', label: t('workflows.name') },
  ],
  [t]
  );

  const DIRECTION_OPTIONS: ComboboxOption[] = useMemo(() => [
    { value: 'ASC', label: t('workflows.ascending') },
    { value: 'DESC', label: t('workflows.descending') },
  ],
  [t]
  );

  const orderByOptions = useMemo(() => {
    return projectId ? ORDER_BY_OPTIONS.filter((option) => option.value !== 'opened_at') : ORDER_BY_OPTIONS;
  }, [projectId]);

  const [order_by, setOrderBy] = useState<WorkflowRecordOrderBy>(orderByOptions[0]?.value as WorkflowRecordOrderBy);
  const [direction, setDirection] = useState<SQLiteDirection>('ASC');
  const [debouncedQuery] = useDebounce(query, 500);

  const queryArg = useMemo<Parameters<typeof useListWorkflowsQuery>[0]>(() => {
    if (selectedCategory !== 'default') {
      return {
        page,
        per_page: PER_PAGE,
        order_by,
        direction,
        category: selectedCategory,
        query: debouncedQuery,
      };
    }
    return {
      page,
      per_page: PER_PAGE,
      order_by: 'name' as const,
      direction: 'ASC' as const,
      category: selectedCategory,
      query: debouncedQuery,
    };
  }, [selectedCategory, debouncedQuery, direction, order_by, page]);

  const { data, isLoading, isError, isFetching } = useListWorkflowsQuery(queryArg);

  const onChangeOrderBy = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isOrderBy(v?.value) || v.value === order_by) {
        return;
      }
      setOrderBy(v.value);
      setPage(0);
    },
    [order_by]
  );
  const valueOrderBy = useMemo(() => orderByOptions.find((o) => o.value === order_by), [order_by, orderByOptions]);

  const onChangeDirection = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isDirection(v?.value) || v.value === direction) {
        return;
      }
      setDirection(v.value);
      setPage(0);
    },
    [direction]
  );
  const valueDirection = useMemo(() => DIRECTION_OPTIONS.find((o) => o.value === direction), [direction]);

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

  const handleChangeFilterText = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
    setPage(0);
  }, []);

  const handleSetCategory = useCallback((category: WorkflowCategory) => {
    setSelectedCategory(category);
    setPage(0);
  }, []);

  return (
    <>
      <Flex gap={4} alignItems="center" h={16} flexShrink={0} flexGrow={0} justifyContent="space-between">
        <ButtonGroup alignSelf="flex-end">
          {workflowCategories.map((category) => (
            <Button
              key={category}
              variant={selectedCategory === category ? undefined : 'ghost'}
              onClick={handleSetCategory.bind(null, category)}
              isChecked={selectedCategory === category}
            >
              {t(`workflows.${category}Workflows`)}
            </Button>
          ))}
        </ButtonGroup>
        {selectedCategory !== 'default' && (
          <>
            <FormControl
              isDisabled={isFetching}
              sx={{
                flexDir: 'column',
                alignItems: 'flex-start',
                gap: 1,
                maxW: 56,
              }}
            >
              <FormLabel>{t('common.orderBy')}</FormLabel>
              <Combobox value={valueOrderBy} options={orderByOptions} onChange={onChangeOrderBy} />
            </FormControl>
            <FormControl
              isDisabled={isFetching}
              sx={{
                flexDir: 'column',
                alignItems: 'flex-start',
                gap: 1,
                maxW: 56,
              }}
            >
              <FormLabel>{t('common.direction')}</FormLabel>
              <Combobox value={valueDirection} options={DIRECTION_OPTIONS} onChange={onChangeDirection} />
            </FormControl>
          </>
        )}
        <InputGroup w="20rem" alignSelf="flex-end">
          <Input
            placeholder={t('workflows.searchWorkflows')}
            value={query}
            onKeyDown={handleKeydownFilterText}
            onChange={handleChangeFilterText}
            data-testid="workflow-search-input"
            minW={64}
          />
          {query.trim().length && (
            <InputRightElement h="full" pe={2}>
              <IconButton
                onClick={resetFilterText}
                size="sm"
                variant="link"
                aria-label={t('workflows.clearWorkflowSearchFilter')}
                icon={<PiXBold />}
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
        <IAINoContentFallback label={t('workflows.noWorkflows')} />
      )}
      <Divider />

      <Flex w="full">
        <Box flex="1">
          <UploadWorkflowButton />
        </Box>
        <Box flex="1" textAlign="center">
          {data && <WorkflowLibraryPagination data={data} page={page} setPage={setPage} />}
        </Box>
        <Box flex="1"></Box>
      </Flex>
    </>
  );
};

export default memo(WorkflowLibraryList);
