import type { ComboboxOnChange, ComboboxOption, SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Combobox, Flex, FormControl, Spinner, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { NO_DRAG_CLASS, NO_FIT_ON_DOUBLE_CLICK_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsInfiniteInfiniteQuery } from 'services/api/endpoints/workflows';
import type { S } from 'services/api/types';

import {
  buildSavedWorkflowOptions,
  getSavedWorkflowSelectionState,
  getSelectedWorkflowOption,
} from './callSavedWorkflowsNodeUtils';
import InvocationNodeHeader from './InvocationNodeHeader';

const MISSING_SELECTION_LABEL = 'Missing or inaccessible workflow';

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
  const dispatch = useAppDispatch();
  const { items, isLoading, isFetching } = useListWorkflowsInfiniteInfiniteQuery(queryArg, queryOptions);
  const selectWorkflowId = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (node?.type !== 'invocation') {
          return '';
        }

        const workflowId = node.data.inputs.workflow_id?.value;
        return typeof workflowId === 'string' ? workflowId : '';
      }),
    [nodeId]
  );
  const workflowId = useAppSelector(selectWorkflowId);
  const options = useMemo(() => buildSavedWorkflowOptions(items), [items]);
  const selectionState = useMemo(() => getSavedWorkflowSelectionState(items, workflowId), [items, workflowId]);
  const selectedOption = useMemo(
    () => getSelectedWorkflowOption(items, workflowId, MISSING_SELECTION_LABEL),
    [items, workflowId]
  );

  const onChange = useCallback<ComboboxOnChange>(
    (value) => {
      dispatch(
        fieldValueReset({
          nodeId,
          fieldName: 'workflow_id',
          value: value?.value ?? '',
        })
      );
    },
    [dispatch, nodeId]
  );

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
          {isLoading ? (
            <LoadingState />
          ) : (
            <WorkflowPicker
              options={options}
              selectionState={selectionState}
              selectedOption={selectedOption}
              isFetching={isFetching}
              onChange={onChange}
            />
          )}
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

const WorkflowPicker = memo(
  ({
    options,
    selectionState,
    selectedOption,
    isFetching,
    onChange,
  }: {
    options: ComboboxOption[];
    selectionState:
      | { status: 'unselected' }
      | { status: 'selected'; workflow: S['WorkflowRecordListItemWithThumbnailDTO'] }
      | { status: 'missing'; workflowId: string };
    selectedOption: ComboboxOption | null;
    isFetching: boolean;
    onChange: ComboboxOnChange;
  }) => {
    const { t } = useTranslation();
    const noOptionsMessage = useCallback(() => t('nodes.noMatchingWorkflows'), [t]);

    if (options.length === 0) {
      return <IAINoContentFallback icon={null} label={t('nodes.noWorkflows')} fontSize="sm" py={4} />;
    }

    return (
      <Flex flexDir="column" gap={1}>
        <FormControl className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS} ${NO_FIT_ON_DOUBLE_CLICK_CLASS}`}>
          <Combobox
            value={selectedOption}
            options={options}
            onChange={onChange}
            placeholder={t('controlLayers.workflowIntegration.selectPlaceholder')}
            noOptionsMessage={noOptionsMessage}
            isClearable
          />
        </FormControl>
        {selectionState.status === 'selected' ? (
          <Flex alignItems="center" gap={1} flexWrap="wrap">
            <Text fontSize="xs" variant="subtext" noOfLines={1}>
              {selectionState.workflow.name}
            </Text>
            {selectionState.workflow.category === 'default' && <Badge variant="subtle">Default</Badge>}
            {selectionState.workflow.is_public && selectionState.workflow.category !== 'default' && (
              <Badge variant="subtle">Shared</Badge>
            )}
          </Flex>
        ) : (
          <SelectionStatusBadge selectionState={selectionState} />
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
WorkflowPicker.displayName = 'WorkflowPicker';

const SelectionStatusBadge = memo(
  ({
    selectionState,
  }: {
    selectionState:
      | { status: 'unselected' }
      | { status: 'selected'; workflow: S['WorkflowRecordListItemWithThumbnailDTO'] }
      | { status: 'missing'; workflowId: string };
  }) => {
    if (selectionState.status === 'selected') {
      return null;
    }

    return (
      <Badge variant="subtle">
        {selectionState.status === 'missing' ? MISSING_SELECTION_LABEL : 'Choose a workflow'}
      </Badge>
    );
  }
);
SelectionStatusBadge.displayName = 'SelectionStatusBadge';
