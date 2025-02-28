import { Flex, Grid, GridItem, Spinner } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import {
  selectWorkflowCategories,
  selectWorkflowOrderBy,
  selectWorkflowOrderDirection,
  selectWorkflowSearchTerm,
} from 'features/nodes/store/workflowSlice';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import { useDebounce } from 'use-debounce';

import { WorkflowLibraryPagination } from './WorkflowLibraryPagination';
import { WorkflowListItem } from './WorkflowListItem';

const PER_PAGE = 6;

export const WorkflowList = () => {
  const searchTerm = useAppSelector(selectWorkflowSearchTerm);
  const { t } = useTranslation();

  const [page, setPage] = useState(0);
  const categories = useAppSelector(selectWorkflowCategories);
  const orderBy = useAppSelector(selectWorkflowOrderBy);
  const direction = useAppSelector(selectWorkflowOrderDirection);
  const query = useAppSelector(selectWorkflowSearchTerm);
  const [debouncedQuery] = useDebounce(query, 500);

  useEffect(() => {
    setPage(0);
  }, [categories, query]);

  const queryArg = useMemo<Parameters<typeof useListWorkflowsQuery>[0]>(() => {
    return {
      page,
      per_page: PER_PAGE,
      order_by: orderBy,
      direction,
      categories,
      query: debouncedQuery,
    };
  }, [direction, orderBy, page, categories, debouncedQuery]);

  const { data, isLoading } = useListWorkflowsQuery(queryArg);

  if (isLoading) {
    return (
      <Flex alignItems="center" justifyContent="center" p={20}>
        <Spinner />
      </Flex>
    );
  }

  if (!data?.items.length) {
    return (
      <IAINoContentFallback
        fontSize="sm"
        py={4}
        label={searchTerm ? t('nodes.noMatchingWorkflows') : t('nodes.noWorkflows')}
        icon={null}
      />
    );
  }

  return (
    <Flex flexDir="column" gap={6}>
      <Grid templateColumns="repeat(2, minmax(200px, 3fr))" templateRows="1fr 1fr 1fr" gap={4}>
        {data?.items.map((workflow) => (
          <GridItem key={workflow.workflow_id}>
            <WorkflowListItem workflow={workflow} key={workflow.workflow_id} />
          </GridItem>
        ))}
      </Grid>
      <WorkflowLibraryPagination page={page} setPage={setPage} data={data} />
    </Flex>
  );
};
