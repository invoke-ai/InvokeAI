import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Badge, Combobox, Flex, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldStringValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { SavedWorkflowFieldInputInstance, SavedWorkflowFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsInfiniteInfiniteQuery } from 'services/api/endpoints/workflows';

import {
  buildSavedWorkflowOptions,
  EMPTY_SELECTION_LABEL,
  getSavedWorkflowSelectionOption,
  getSavedWorkflowSelectionState,
  getSavedWorkflowSelectionStatusLabel,
} from './savedWorkflowFieldUtils';
import type { FieldComponentProps } from './types';

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

const SavedWorkflowFieldInputComponent = (
  props: FieldComponentProps<SavedWorkflowFieldInputInstance, SavedWorkflowFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { items, isLoading, isFetching } = useListWorkflowsInfiniteInfiniteQuery(queryArg, queryOptions);

  const options = useMemo<ComboboxOption[]>(() => buildSavedWorkflowOptions(items), [items]);
  const selectionState = useMemo(() => getSavedWorkflowSelectionState(items, field.value), [field.value, items]);
  const value = useMemo<ComboboxOption | null>(() => getSavedWorkflowSelectionOption(selectionState), [selectionState]);
  const statusLabel = useMemo(() => getSavedWorkflowSelectionStatusLabel(selectionState), [selectionState]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      dispatch(
        fieldStringValueChanged({
          nodeId,
          fieldName: field.name,
          value: v?.value ?? '',
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const noOptionsMessage = useCallback(() => t('nodes.noMatchingWorkflows'), [t]);

  return (
    <Flex flexDir="column" gap={1}>
      <Combobox
        className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
        value={value}
        options={options}
        onChange={onChange}
        placeholder={isLoading ? t('common.loading') : t('controlLayers.workflowIntegration.selectPlaceholder')}
        noOptionsMessage={noOptionsMessage}
        isClearable
      />
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
        <Badge variant="subtle">{statusLabel ?? EMPTY_SELECTION_LABEL}</Badge>
      )}
      {isFetching && (
        <Text variant="subtext" fontSize="xs">
          Updating...
        </Text>
      )}
    </Flex>
  );
};

export default memo(SavedWorkflowFieldInputComponent);
