import { Button, Flex, Grid, GridItem, Spacer, Spinner } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import {
  selectWorkflowLibrarySelectedTags,
  selectWorkflowOrderBy,
  selectWorkflowOrderDirection,
  selectWorkflowSearchTerm,
  selectWorkflowSelectedCategories,
} from 'features/nodes/store/workflowSlice';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import type { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import { useListWorkflowsInfiniteInfiniteQuery } from 'services/api/endpoints/workflows';
import type { S } from 'services/api/types';
import { useDebounce } from 'use-debounce';

import { WorkflowListItem } from './WorkflowListItem';

const PER_PAGE = 30;

const useInfiniteQueryAry = () => {
  const categories = useAppSelector(selectWorkflowSelectedCategories);
  const orderBy = useAppSelector(selectWorkflowOrderBy);
  const direction = useAppSelector(selectWorkflowOrderDirection);
  const query = useAppSelector(selectWorkflowSearchTerm);
  const tags = useAppSelector(selectWorkflowLibrarySelectedTags);
  const [debouncedQuery] = useDebounce(query, 500);

  const queryArg = useMemo(() => {
    return {
      page: 0,
      per_page: PER_PAGE,
      order_by: orderBy ?? 'opened_at',
      direction,
      categories,
      query: debouncedQuery,
      tags: categories.length === 1 && categories.includes('default') ? tags : [],
    } satisfies Parameters<typeof useListWorkflowsQuery>[0];
  }, [orderBy, direction, categories, debouncedQuery, tags]);

  return queryArg;
};

const queryOptions = {
  selectFromResult: ({ data, ...rest }) => {
    return {
      items: data?.pages.map(({ items }) => items).flat() ?? EMPTY_ARRAY,
      ...rest,
    } as const;
  },
} satisfies Parameters<typeof useListWorkflowsInfiniteInfiniteQuery>[1];

export const WorkflowList = () => {
  const searchTerm = useAppSelector(selectWorkflowSearchTerm);
  const { t } = useTranslation();
  const queryArg = useInfiniteQueryAry();
  const { items, isFetching, isLoading, fetchNextPage, hasNextPage } = useListWorkflowsInfiniteInfiniteQuery(
    queryArg,
    queryOptions
  );

  if (isLoading) {
    return (
      <Flex alignItems="center" justifyContent="center" w="full" h="full">
        <Spinner />
      </Flex>
    );
  }

  if (items.length === 0) {
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
    <WorkflowListContent
      items={items}
      hasNextPage={hasNextPage}
      fetchNextPage={fetchNextPage}
      isFetching={isFetching}
    />
  );
};

const WorkflowListContent = memo(
  ({
    items,
    hasNextPage,
    isFetching,
    fetchNextPage,
  }: {
    items: S['WorkflowRecordListItemWithThumbnailDTO'][];
    hasNextPage: boolean;
    isFetching: boolean;
    fetchNextPage: ReturnType<typeof useListWorkflowsInfiniteInfiniteQuery>['fetchNextPage'];
  }) => {
    const { t } = useTranslation();
    const ref = useRef<HTMLDivElement>(null);

    const onScroll = useCallback(() => {
      if (!hasNextPage || isFetching) {
        return;
      }
      const el = ref.current;
      if (!el) {
        return;
      }
      const { scrollTop, scrollHeight, clientHeight } = el;
      if (Math.abs(scrollHeight - (scrollTop + clientHeight)) <= 1) {
        fetchNextPage();
      }
    }, [hasNextPage, isFetching, fetchNextPage]);

    const loadMore = useCallback(() => {
      if (!hasNextPage || isFetching) {
        return;
      }
      const el = ref.current;
      if (!el) {
        return;
      }
      fetchNextPage();
    }, [hasNextPage, isFetching, fetchNextPage]);

    // // TODO(psyche): this causes an infinite loop, the scrollIntoView triggers the onScroll which triggers the
    // // fetchNextPage which triggers the scrollIntoView again...
    // useEffect(() => {
    //   const el = ref.current;
    //   if (!el) {
    //     return;
    //   }

    //   const observer = new MutationObserver(() => {
    //     el.querySelector(':scope > :last-child')?.scrollIntoView({ behavior: 'smooth' });
    //   });

    //   observer.observe(el, { childList: true });

    //   return () => {
    //     observer.disconnect();
    //   };
    // }, []);

    return (
      <Flex flexDir="column" gap={4} flex={1} minH={0}>
        <Grid
          ref={ref}
          templateColumns="repeat(auto-fill, minmax(340px, 1fr))"
          gridAutoFlow="dense"
          gap={4}
          overflow="scroll"
          onScroll={onScroll}
        >
          {items.map((workflow) => (
            <GridItem id={`grid-${workflow.workflow_id}`} key={workflow.workflow_id}>
              <WorkflowListItem workflow={workflow} key={workflow.workflow_id} />
            </GridItem>
          ))}
        </Grid>
        <Spacer />
        {hasNextPage && (
          <Button
            onClick={loadMore}
            isLoading={isFetching}
            loadingText={t('common.loading')}
            variant="ghost"
            w="min-content"
            alignSelf="center"
          >
            {t('workflows.loadMore')}
          </Button>
        )}
      </Flex>
    );
  }
);
WorkflowListContent.displayName = 'WorkflowListContent';
