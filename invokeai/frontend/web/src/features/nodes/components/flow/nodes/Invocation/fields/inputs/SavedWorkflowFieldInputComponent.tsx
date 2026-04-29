import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Badge, Combobox, Flex, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldStringValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { SavedWorkflowFieldInputInstance, SavedWorkflowFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetWorkflowQuery, useListWorkflowsInfiniteInfiniteQuery } from 'services/api/endpoints/workflows';

import {
  buildSavedWorkflowOptions,
  getSavedWorkflowDisplayState,
  getSavedWorkflowListItemFromRecord,
  getSavedWorkflowSelectionOption,
  getSavedWorkflowSelectionState,
  MISSING_WORKFLOW_OPTION_VALUE,
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
  const isSelectedWorkflowInList = useMemo(
    () => items.some((workflow) => workflow.workflow_id === field.value),
    [field.value, items]
  );
  const { data: selectedWorkflowRecord } = useGetWorkflowQuery(field.value, {
    skip: !field.value || isSelectedWorkflowInList,
  });
  const selectedWorkflow = useMemo(
    () => (selectedWorkflowRecord ? getSavedWorkflowListItemFromRecord(selectedWorkflowRecord) : undefined),
    [selectedWorkflowRecord]
  );

  const options = useMemo<ComboboxOption[]>(() => buildSavedWorkflowOptions(items), [items]);
  const selectionState = useMemo(
    () => getSavedWorkflowSelectionState(items, field.value, selectedWorkflow),
    [field.value, items, selectedWorkflow]
  );
  const value = useMemo<ComboboxOption | null>(() => {
    const option = getSavedWorkflowSelectionOption(selectionState);
    if (option?.value === MISSING_WORKFLOW_OPTION_VALUE) {
      return {
        ...option,
        label: t('nodes.savedWorkflowMissing'),
      };
    }
    return option;
  }, [selectionState, t]);
  const statusLabel = useMemo(() => {
    const displayState = getSavedWorkflowDisplayState(selectionState);
    return displayState.statusLabelKey ? t(displayState.statusLabelKey) : null;
  }, [selectionState, t]);
  const displayState = useMemo(() => getSavedWorkflowDisplayState(selectionState), [selectionState]);

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
          {displayState.badges.includes('unsupported') && (
            <Badge variant="subtle">{t('nodes.savedWorkflowUnsupported')}</Badge>
          )}
          {displayState.badges.includes('default') && (
            <Badge variant="subtle">{t('nodes.savedWorkflowDefaultBadge')}</Badge>
          )}
          {displayState.badges.includes('shared') && <Badge variant="subtle">{t('workflows.shared')}</Badge>}
        </Flex>
      ) : (
        <Badge variant="subtle">{statusLabel ?? t('nodes.savedWorkflowChoose')}</Badge>
      )}
      {selectionState.status === 'selected' && displayState.compatibilityMessage && (
        <Text variant="subtext" fontSize="xs">
          {displayState.compatibilityMessage}
        </Text>
      )}
      {isFetching && (
        <Text variant="subtext" fontSize="xs">
          {t('nodes.savedWorkflowUpdating')}
        </Text>
      )}
    </Flex>
  );
};

export default memo(SavedWorkflowFieldInputComponent);
