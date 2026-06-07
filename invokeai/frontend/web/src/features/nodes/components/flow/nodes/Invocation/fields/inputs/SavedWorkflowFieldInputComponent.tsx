import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Badge, Combobox, Flex, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldStringValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { SavedWorkflowFieldInputInstance, SavedWorkflowFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useDeferredValue, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetWorkflowQuery, useListWorkflowsInfiniteInfiniteQuery } from 'services/api/endpoints/workflows';

import {
  buildSavedWorkflowOptions,
  getSavedWorkflowDisplayState,
  getSavedWorkflowListItemFromRecord,
  getSavedWorkflowPickerOwnedQueryArg,
  getSavedWorkflowPickerSharedQueryArg,
  getSavedWorkflowSelectionOption,
  getSavedWorkflowSelectionState,
  mergeSavedWorkflowPickerItems,
  MISSING_WORKFLOW_OPTION_VALUE,
  shouldFetchNextSavedWorkflowPickerPage,
} from './savedWorkflowFieldUtils';
import type { FieldComponentProps } from './types';

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
  const [workflowSearchQuery, setWorkflowSearchQuery] = useState('');
  const deferredWorkflowSearchQuery = useDeferredValue(workflowSearchQuery);
  const ownedQueryArg = useMemo(
    () =>
      getSavedWorkflowPickerOwnedQueryArg(deferredWorkflowSearchQuery) satisfies Parameters<
        typeof useListWorkflowsInfiniteInfiniteQuery
      >[0],
    [deferredWorkflowSearchQuery]
  );
  const sharedQueryArg = useMemo(
    () =>
      getSavedWorkflowPickerSharedQueryArg(deferredWorkflowSearchQuery) satisfies Parameters<
        typeof useListWorkflowsInfiniteInfiniteQuery
      >[0],
    [deferredWorkflowSearchQuery]
  );
  const {
    items: ownedItems,
    isLoading: isOwnedLoading,
    isFetching: isOwnedFetching,
    hasNextPage: hasNextOwnedPage,
    fetchNextPage: fetchNextOwnedPage,
  } = useListWorkflowsInfiniteInfiniteQuery(ownedQueryArg, queryOptions);
  const {
    items: sharedItems,
    isLoading: isSharedLoading,
    isFetching: isSharedFetching,
    hasNextPage: hasNextSharedPage,
    fetchNextPage: fetchNextSharedPage,
  } = useListWorkflowsInfiniteInfiniteQuery(sharedQueryArg, queryOptions);
  const items = useMemo(() => mergeSavedWorkflowPickerItems(ownedItems, sharedItems), [ownedItems, sharedItems]);
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
  const onMenuScrollToBottom = useCallback(() => {
    if (shouldFetchNextSavedWorkflowPickerPage({ hasNextPage: hasNextOwnedPage, isFetching: isOwnedFetching })) {
      fetchNextOwnedPage();
    }
    if (shouldFetchNextSavedWorkflowPickerPage({ hasNextPage: hasNextSharedPage, isFetching: isSharedFetching })) {
      fetchNextSharedPage();
    }
  }, [fetchNextOwnedPage, fetchNextSharedPage, hasNextOwnedPage, hasNextSharedPage, isOwnedFetching, isSharedFetching]);
  const onInputChange = useCallback((inputValue: string) => {
    setWorkflowSearchQuery(inputValue);
    return inputValue;
  }, []);

  const noOptionsMessage = useCallback(() => t('nodes.noMatchingWorkflows'), [t]);
  const isLoading = isOwnedLoading || isSharedLoading;
  const isFetching = isOwnedFetching || isSharedFetching;

  return (
    <Flex flexDir="column" gap={1}>
      <Combobox
        className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
        value={value}
        options={options}
        onChange={onChange}
        onInputChange={onInputChange}
        onMenuScrollToBottom={onMenuScrollToBottom}
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
