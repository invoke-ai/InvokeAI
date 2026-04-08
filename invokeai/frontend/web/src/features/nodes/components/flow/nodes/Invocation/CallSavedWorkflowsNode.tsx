import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Flex, Spinner, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo, useMemo } from 'react';
import { useListWorkflowsInfiniteInfiniteQuery } from 'services/api/endpoints/workflows';
import type { S } from 'services/api/types';

import InvocationNodeHeader from './InvocationNodeHeader';

type Props = {
  nodeId: string;
  isOpen: boolean;
};

const bodySx: SystemStyleObject = {
  flexDirection: 'column',
  w: 'full',
  h: 'full',
  py: 2,
  gap: 2,
  borderBottomRadius: 'base',
  '&[data-is-open="false"]': {
    display: 'none',
  },
};

const queryArg = {
  page: 0,
  per_page: 50,
  order_by: 'name',
  direction: 'ASC',
  categories: ['user', 'default'],
  query: '',
  tags: [],
  has_been_opened: undefined,
  is_public: undefined,
} satisfies Parameters<typeof useListWorkflowsInfiniteInfiniteQuery>[0];

const queryOptions = {
  selectFromResult: ({ data, ...rest }) => ({
    items: data?.pages.flatMap(({ items }) => items) ?? EMPTY_ARRAY,
    ...rest,
  }),
} satisfies Parameters<typeof useListWorkflowsInfiniteInfiniteQuery>[1];

const CallSavedWorkflowsNode = ({ nodeId, isOpen }: Props) => {
  const { items, isLoading, isFetching } = useListWorkflowsInfiniteInfiniteQuery(queryArg, queryOptions);

  return (
    <>
      <InvocationNodeHeader nodeId={nodeId} isOpen={isOpen} />
      <Flex layerStyle="nodeBody" sx={bodySx} data-is-open={isOpen}>
        <Flex flexDir="column" px={2} gap={2} w="full">
          <Flex alignItems="center" justifyContent="space-between">
            <Text fontSize="sm" fontWeight="semibold">
              Saved Workflows
            </Text>
            <Badge variant="subtle">{items.length}</Badge>
          </Flex>
          {isLoading ? <LoadingState /> : <WorkflowItems items={items} isFetching={isFetching} />}
        </Flex>
      </Flex>
    </>
  );
};

export default memo(CallSavedWorkflowsNode);

const LoadingState = memo(() => {
  return (
    <Flex alignItems="center" justifyContent="center" minH={24}>
      <Spinner size="sm" />
    </Flex>
  );
});
LoadingState.displayName = 'LoadingState';

const WorkflowItems = memo(
  ({ items, isFetching }: { items: S['WorkflowRecordListItemWithThumbnailDTO'][]; isFetching: boolean }) => {
    const visibleItems = useMemo(() => items.slice(0, 8), [items]);

    if (visibleItems.length === 0) {
      return <IAINoContentFallback icon={null} label="No saved workflows" fontSize="sm" py={4} />;
    }

    return (
      <Flex flexDir="column" gap={1}>
        {visibleItems.map((workflow) => (
          <Flex
            key={workflow.workflow_id}
            alignItems="center"
            justifyContent="space-between"
            gap={2}
            borderRadius="base"
            bg="base.800"
            px={2}
            py={1.5}
          >
            <Text fontSize="sm" noOfLines={1}>
              {workflow.name}
            </Text>
            <Flex alignItems="center" gap={1} flexShrink={0}>
              {workflow.category === 'default' && <Badge variant="subtle">Default</Badge>}
              {workflow.is_public && workflow.category !== 'default' && <Badge variant="subtle">Shared</Badge>}
            </Flex>
          </Flex>
        ))}
        {items.length > visibleItems.length && (
          <Text variant="subtext" fontSize="xs">
            Showing {visibleItems.length} of {items.length} workflows
          </Text>
        )}
        {isFetching && (
          <Text variant="subtext" fontSize="xs">
            Updating...
          </Text>
        )}
      </Flex>
    );
  }
);
WorkflowItems.displayName = 'WorkflowItems';
