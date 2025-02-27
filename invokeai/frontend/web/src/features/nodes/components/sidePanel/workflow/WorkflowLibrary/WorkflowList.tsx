import { Flex, Grid, GridItem, Spinner } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import type { WorkflowLibraryCategory } from 'features/nodes/store/types';
import {
  selectWorkflowBrowsingCategory,
  selectWorkflowOrderBy,
  selectWorkflowOrderDirection,
  selectWorkflowSearchTerm,
} from 'features/nodes/store/workflowSlice';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import { useDebounce } from 'use-debounce';

import { WorkflowLibraryPagination } from './WorkflowLibraryPagination';
import { WorkflowListItem } from './WorkflowListItem';

const PER_PAGE = 6;

const mapUiCategoryToApiCategory = (sideNav: WorkflowLibraryCategory): WorkflowCategory[] => {
  switch (sideNav) {
    case 'account':
      return ['user', 'project'];
    case 'private':
      return ['user'];
    case 'shared':
      return ['project'];
    case 'default':
      return ['default'];
    default:
      return [];
  }
};

export const WorkflowList = () => {
  const searchTerm = useAppSelector(selectWorkflowSearchTerm);
  const { t } = useTranslation();

  const [page, setPage] = useState(0);
  const browsingCategory = useAppSelector(selectWorkflowBrowsingCategory);
  const orderBy = useAppSelector(selectWorkflowOrderBy);
  const direction = useAppSelector(selectWorkflowOrderDirection);
  const query = useAppSelector(selectWorkflowSearchTerm);
  const [debouncedQuery] = useDebounce(query, 500);

  useEffect(() => {
    setPage(0);
  }, [browsingCategory, query]);

  const queryArg = useMemo<Parameters<typeof useListWorkflowsQuery>[0]>(() => {
    const categories = mapUiCategoryToApiCategory(browsingCategory);
    return {
      page,
      per_page: PER_PAGE,
      order_by: orderBy,
      direction,
      categories,
      query: debouncedQuery,
    };
  }, [direction, orderBy, page, browsingCategory, debouncedQuery]);

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
      <Grid templateColumns="repeat(2, minmax(200px, 3fr))" gap={4}>
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
