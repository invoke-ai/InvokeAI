import type { IntermediatesRow } from '@features/intermediates/core/types';
/* eslint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { IntermediatesSummaryParams } from '@features/intermediates/data/keys';
import type { TFunction } from 'i18next';

import { Box, HStack, Icon, Input, InputGroup, Spinner, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { isRowSelected, resolveScope } from '@features/intermediates/core/selection';
import { isOperationSettled } from '@features/intermediates/core/types';
import { consumeIntermediatesFocus, peekIntermediatesFocus } from '@features/intermediates/data/focus';
import { intermediatesKeys } from '@features/intermediates/data/keys';
import { activeOperationStore, reconcileIntermediatesOperations } from '@features/intermediates/data/operationStore';
import { INTERMEDIATES_PAGE_SIZE, intermediatesSummaryQueryOptions } from '@features/intermediates/data/queries';
import { formatBytes } from '@platform/i18n/languages';
import { useMountEffect } from '@platform/react/useMountEffect';
import { ApiError, getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { ListPager } from '@platform/ui/list/ListPager';
import { ListSelectionBar } from '@platform/ui/list/ListSelectionBar';
import { RemovableTag } from '@platform/ui/RemovableTag';
import { Tooltip } from '@platform/ui/Tooltip';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { BrushCleaningIcon, RefreshCwIcon, SearchIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ClearDialog } from './ClearDialog';
import { getOwnerLabel, IntermediatesList } from './IntermediatesList';
import { getOperationAnnouncement, OperationPanel } from './OperationPanel';
import { useCleanupDialog } from './useCleanupDialog';
import { useFollowedOperation } from './useFollowedOperation';
import { useIntermediatesSelection } from './useIntermediatesSelection';

export interface IntermediatesManagerProps {
  /** The current account, or null in single-user mode where the install is the only account. */
  currentUserId: string | null;
  /** The current account's display name or email, for the account filter before its rows load. */
  currentUserLabel?: string | null;
  canClearOthersIntermediates: boolean;
}

const SEARCH_ICON = <Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />;
const EMPTY_ROWS: readonly IntermediatesRow[] = [];
const SEARCH_DEBOUNCE_MS = 250;

/** Paging and refining the search keep the current rows on screen, inert, until the new ones arrive. */
const isSameListExceptPageOrSearch = (previous: unknown, next: IntermediatesSummaryParams): boolean => {
  if (typeof previous !== 'object' || previous === null) {
    return false;
  }
  const { offset: _previousOffset, search: _previousSearch, ...previousRest } = previous as IntermediatesSummaryParams;
  const { offset: _nextOffset, search: _nextSearch, ...nextRest } = next;
  const keys = new Set([...Object.keys(previousRest), ...Object.keys(nextRest)]) as Set<keyof typeof nextRest>;
  return [...keys].every((key) => previousRest[key] === nextRest[key]);
};

const formatSummarySize = (bytes: number, unknownCount: number, t: TFunction): string =>
  unknownCount > 0
    ? `${formatBytes(bytes)} ${t('intermediates.list.unmeasured', { count: unknownCount })}`
    : formatBytes(bytes);

/** The Settings section: search, a select-all row with the delete action, and one row per project. */
export const IntermediatesManager = ({
  canClearOthersIntermediates,
  currentUserId,
  currentUserLabel = null,
}: IntermediatesManagerProps) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  // An entry point's intent is read when the section first renders and consumed once that render commits.
  const [focus] = useState(peekIntermediatesFocus);
  const [search, setSearch] = useState('');
  const [querySearch, setQuerySearch] = useState('');
  const searchTimerRef = useRef<number | undefined>(undefined);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const refreshRequestRef = useRef(0);
  const deleteReasonId = useId();
  useMountEffect(() => {
    consumeIntermediatesFocus(focus);
    return () => window.clearTimeout(searchTimerRef.current);
  });
  const [ownerFilter, setOwnerFilter] = useState<string | null>(() =>
    focus?.ownerId && canClearOthersIntermediates ? focus.ownerId : currentUserId
  );
  const [projectFilter, setProjectFilter] = useState<string | null>(focus?.projectId ?? null);
  const [offset, setOffset] = useState(0);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const operationRegionRef = useRef<HTMLDivElement | null>(null);
  const operation = useFollowedOperation({ fallbackFocusRef: searchRef });
  // Operations live in server memory; after a restart the server no longer knows this one, and asking again cannot help.
  const isLookupGone =
    operation.query.isError && operation.query.error instanceof ApiError && operation.query.error.status === 404;
  const lookupError = !operation.query.isError
    ? null
    : isLookupGone
      ? t('intermediates.operation.lookupGone')
      : getApiErrorMessage(operation.query.error, t('intermediates.operation.lookupFailed'));
  // The live region stays mounted and starts empty; screen readers skip text that arrives with its region.
  const announcement = getOperationAnnouncement(
    operation.operationId ? (operation.query.data ?? null) : null,
    operation.operationId !== null && operation.query.isPending,
    t,
    operation.operationId ? lookupError : null
  );
  const [announced, setAnnounced] = useState({ source: announcement, text: '' });
  if (announced.source !== announcement) {
    setAnnounced({ source: announcement, text: announcement });
  }

  // Non-admins are confined to their own rows by the server whatever is sent.
  const ownerId: string | null = canClearOthersIntermediates ? ownerFilter : currentUserId;
  const params = useMemo<IntermediatesSummaryParams>(
    () => ({
      limit: INTERMEDIATES_PAGE_SIZE,
      offset,
      order: 'desc',
      ownerId,
      projectId: projectFilter,
      search: querySearch,
      sort: 'reclaimable_bytes',
    }),
    [offset, ownerId, projectFilter, querySearch]
  );
  const query = useQuery({
    ...intermediatesSummaryQueryOptions(params),
    placeholderData: (previous, previousQuery) =>
      previousQuery && isSameListExceptPageOrSearch(previousQuery.queryKey.at(-1), params) ? previous : undefined,
  });
  // Rows on screen belong to an older request while a page or search loads, or while typing has not settled.
  const isListBusy = query.isPlaceholderData || search !== querySearch;
  const rows = query.data?.items ?? EMPTY_ROWS;
  const totals = query.data?.totals;
  const hasSearch = querySearch.trim().length > 0;
  // Rows disappearing (a cleanup, a narrower search) can leave the offset past the end; step back to a valid page.
  if (query.data && !query.isPlaceholderData && offset > 0 && offset >= query.data.total) {
    setOffset(
      Math.max(0, Math.floor(Math.max(query.data.total - 1, 0) / INTERMEDIATES_PAGE_SIZE) * INTERMEDIATES_PAGE_SIZE)
    );
  }
  const selection = useIntermediatesSelection({
    initialProjectId: focus?.projectId ?? null,
    rows,
    summary: query.data,
  });
  const dialog = useCleanupDialog({ onStarted: selection.reset });
  const { reset: resetSelection } = selection;

  const commitSearch = useCallback(
    (value: string) => {
      window.clearTimeout(searchTimerRef.current);
      setQuerySearch(value);
      setProjectFilter(null);
      setOffset(0);
      // A filter change hides rows; hidden selections would act on what the user can no longer see.
      resetSelection();
    },
    [resetSelection]
  );
  const handleSearchChange = useCallback(
    (value: string) => {
      setSearch(value);
      window.clearTimeout(searchTimerRef.current);
      searchTimerRef.current = window.setTimeout(() => commitSearch(value), SEARCH_DEBOUNCE_MS);
    },
    [commitSearch]
  );
  const clearSearch = useCallback(() => {
    setSearch('');
    commitSearch('');
  }, [commitSearch]);
  const clearOwnerFilter = useCallback(() => {
    setOwnerFilter(null);
    setOffset(0);
    resetSelection();
  }, [resetSelection]);
  const showOwnAccount = useCallback(() => {
    setOwnerFilter(currentUserId);
    setOffset(0);
    resetSelection();
  }, [currentUserId, resetSelection]);
  const clearProjectFilter = useCallback(() => {
    setProjectFilter(null);
    setOffset(0);
    resetSelection();
  }, [resetSelection]);
  // Another press restarts the reload, so the control never locks on a request that does not settle.
  const handleRefresh = useCallback(() => {
    const request = ++refreshRequestRef.current;
    setIsRefreshing(true);
    // A refresh also asks whether a run this tab does not know about is in progress (a start whose answer was lost).
    const catchUp =
      activeOperationStore.getSnapshot().operationId === null
        ? reconcileIntermediatesOperations(queryClient).catch(() => undefined)
        : Promise.resolve();
    void Promise.all([queryClient.invalidateQueries({ queryKey: intermediatesKeys.all }), catchUp]).finally(() => {
      if (refreshRequestRef.current === request) {
        setIsRefreshing(false);
      }
    });
  }, [queryClient]);

  const { selectionSummary } = selection;
  const filteredOwnerLabel = !ownerFilter
    ? null
    : rows[0]?.userId === ownerFilter
      ? getOwnerLabel(rows[0], t)
      : ownerFilter === focus?.ownerId && focus.ownerLabel
        ? focus.ownerLabel
        : ownerFilter === currentUserId && currentUserLabel
          ? currentUserLabel
          : t('intermediates.owner.unknownAccount');
  const hasPreviousPage = offset > 0;
  const hasNextPage = query.data !== undefined && offset + rows.length < query.data.total;
  const showPagination = offset > 0 || (query.data?.total ?? 0) > INTERMEDIATES_PAGE_SIZE;
  const followed = operation.query.data;
  // A second cleanup would replace the panel reporting the first; wait until it settles (or cannot be looked up).
  const isOperationActive =
    operation.operationId !== null && !operation.query.isError && !(followed && isOperationSettled(followed));
  const isDeleteUnavailable = !selection.hasSelection || isOperationActive;
  const goToPage = (nextOffset: number) => {
    if (!isListBusy && nextOffset >= 0 && (nextOffset < offset || hasNextPage)) {
      setOffset(nextOffset);
    }
  };

  return (
    <Stack gap="3" h="full" minH="0">
      <HStack align="flex-start" gap="3" justify="space-between">
        <Stack gap="0.5" minW="0">
          <Text fontSize="sm" fontWeight="600">
            {t('intermediates.title')}
          </Text>
          <Text color="fg.muted" fontSize="xs">
            {t('intermediates.description')}
          </Text>
        </Stack>
        <Tooltip content={t('intermediates.refresh')}>
          <IconButton
            aria-busy={isRefreshing || undefined}
            aria-label={t('intermediates.refresh')}
            flexShrink={0}
            size="xs"
            variant="outline"
            onClick={handleRefresh}
          >
            {isRefreshing ? <Spinner size="xs" /> : <RefreshCwIcon />}
          </IconButton>
        </Tooltip>
      </HStack>
      <HStack flexWrap="wrap" gap="2">
        <InputGroup flex="1 1 16rem" minW="0" startElement={SEARCH_ICON}>
          <Input
            ref={searchRef}
            aria-label={t('intermediates.searchLabel')}
            placeholder={
              canClearOthersIntermediates
                ? t('intermediates.searchPlaceholderAdmin')
                : t('intermediates.searchPlaceholder')
            }
            size="xs"
            value={search}
            onChange={(event) => handleSearchChange(event.currentTarget.value)}
          />
        </InputGroup>
        {canClearOthersIntermediates && filteredOwnerLabel ? (
          <RemovableTag removeLabel={t('intermediates.owner.showEveryone')} onRemove={clearOwnerFilter}>
            {t('intermediates.owner.filtered', { name: filteredOwnerLabel })}
          </RemovableTag>
        ) : null}
        {canClearOthersIntermediates && !ownerFilter && currentUserId ? (
          <Button size="2xs" variant="outline" onClick={showOwnAccount}>
            {t('intermediates.owner.showMine')}
          </Button>
        ) : null}
        {projectFilter ? (
          <RemovableTag removeLabel={t('intermediates.owner.showAllProjects')} onRemove={clearProjectFilter}>
            {rows.find((row) => row.projectId === projectFilter)?.projectName ?? t('intermediates.owner.projectFilter')}
          </RemovableTag>
        ) : null}
      </HStack>
      <VisuallyHidden aria-atomic="true" aria-live="polite" role="status">
        {announced.text}
      </VisuallyHidden>
      {operation.operationId ? (
        // Takes focus after a start, since the Delete control that opened the dialog has nothing left to act on.
        <Box
          ref={operationRegionRef}
          aria-label={t('intermediates.operation.regionLabel')}
          focusVisibleRing="outside"
          role="group"
          rounded="l2"
          tabIndex={-1}
        >
          <OperationPanel
            isLoading={operation.query.isPending}
            isLookupRetryable={!isLookupGone}
            isRefetching={operation.query.isFetching}
            lookupError={lookupError}
            onRefetch={() => void operation.query.refetch()}
            operation={operation.query.data ?? null}
            onDismiss={operation.dismiss}
            onRunAgain={(trigger) => {
              if (followed) {
                dialog.open(followed.scope, trigger, followed.mode);
              }
            }}
          />
        </Box>
      ) : null}
      <Stack flex="1" gap="0" minH="0" mt="-3">
        <ListSelectionBar
          checked={
            selection.selectionState === 'all' ? true : selection.selectionState === 'some' ? 'indeterminate' : false
          }
          isDisabled={rows.length === 0 || isListBusy}
          label={t('intermediates.list.selectAll')}
          summary={
            isOperationActive && selection.hasSelection
              ? t('intermediates.list.deleteWaiting')
              : selection.hasSelection
                ? t('intermediates.selection.estimate', {
                    count: selectionSummary.rows,
                    images: t('intermediates.counts.images', { count: selectionSummary.safeImages }),
                    size: formatSummarySize(selectionSummary.reclaimableBytes, selectionSummary.unknownSizeCount, t),
                    videos: t('intermediates.counts.videos', { count: selectionSummary.safeVideos }),
                  })
                : totals
                  ? t('intermediates.selection.available', {
                      images: t('intermediates.counts.images', { count: totals.safeImages }),
                      size: formatSummarySize(totals.reclaimableBytes, totals.unknownSizeCount, t),
                      videos: t('intermediates.counts.videos', { count: totals.safeVideos }),
                    })
                  : ''
          }
          summaryId={deleteReasonId}
          onCheckedChange={selection.toggleAll}
        >
          {query.data?.measuring ? (
            <Tooltip content={t('intermediates.stats.measuringNote')}>
              <Spinner aria-label={t('intermediates.stats.measuringNote')} color="fg.muted" size="xs" />
            </Tooltip>
          ) : null}
          {/* `aria-disabled` keeps it focusable, so the reason it waits is announced on focus. */}
          <Button
            aria-describedby={isOperationActive && selection.hasSelection ? deleteReasonId : undefined}
            aria-disabled={isDeleteUnavailable}
            colorPalette="red"
            size="2xs"
            variant="ghost"
            onClick={(event) => {
              if (isDeleteUnavailable) {
                return;
              }
              dialog.open(
                resolveScope({
                  ownerId,
                  projectId: projectFilter,
                  search: querySearch,
                  selection: selection.effectiveSelection,
                }),
                event.currentTarget
              );
            }}
          >
            <Icon as={Trash2Icon} boxSize="3" />
            {t('intermediates.list.delete')}
          </Button>
        </ListSelectionBar>
        <IntermediatesList
          emptyState={
            <EmptyState
              description={
                hasSearch ? t('intermediates.empty.noMatchesDescription') : t('intermediates.empty.description')
              }
              icon={<Icon as={hasSearch ? SearchIcon : BrushCleaningIcon} />}
              title={hasSearch ? t('intermediates.empty.noMatches') : t('intermediates.empty.title')}
            >
              {hasSearch ? (
                <Button size="xs" variant="outline" onClick={clearSearch}>
                  {t('common.clearSearch')}
                </Button>
              ) : null}
            </EmptyState>
          }
          errorState={
            <EmptyState
              danger
              description={
                query.isError ? getApiErrorMessage(query.error, t('intermediates.errors.couldNotLoad')) : null
              }
              title={t('intermediates.errors.couldNotLoad')}
            >
              <Button size="xs" variant="outline" onClick={() => void query.refetch()}>
                {t('common.retry')}
              </Button>
            </EmptyState>
          }
          isBusy={isListBusy}
          isSelected={(row) => isRowSelected(selection.effectiveSelection, row)}
          rows={rows}
          showOwner={canClearOthersIntermediates && ownerId === null}
          status={query.isPending ? 'loading' : query.isError ? 'error' : 'ready'}
          onToggleRow={selection.toggleRow}
        />
        {showPagination ? (
          <ListPager
            hasNext={hasNextPage}
            hasPrevious={hasPreviousPage}
            isBusy={isListBusy}
            page={Math.floor((query.data?.offset ?? offset) / INTERMEDIATES_PAGE_SIZE) + 1}
            onNext={() => goToPage(offset + INTERMEDIATES_PAGE_SIZE)}
            onPrevious={() => goToPage(offset - INTERMEDIATES_PAGE_SIZE)}
          />
        ) : null}
      </Stack>
      <ClearDialog
        canManageEveryone={canClearOthersIntermediates}
        currentUserId={currentUserId}
        getFinalFocus={() =>
          (dialog.hasStarted() ? operationRegionRef.current : dialog.triggerRef.current) ?? searchRef.current
        }
        state={dialog.state}
        onClose={dialog.close}
        onConfirm={() => void dialog.confirm()}
        onModeChange={dialog.changeMode}
        onRetryPreview={dialog.retryPreview}
      />
    </Stack>
  );
};
