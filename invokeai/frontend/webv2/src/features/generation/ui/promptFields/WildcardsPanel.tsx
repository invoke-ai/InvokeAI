/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { WildcardRecord } from '@features/generation/data/wildcards';
import type { WildcardCatalog } from '@features/generation/ui/useWildcards';
import type { ListRowProps } from '@platform/ui/list/List';
import type { ListRow } from '@platform/ui/list/listRows';
import type { ChangeEvent } from 'react';

import { Box, Field as ChakraField, HStack, Input, Stack, Text } from '@chakra-ui/react';
import {
  getWildcardNameError,
  getWildcardValuesError,
  normalizeWildcardValues,
} from '@features/generation/core/dynamicPrompts';
import { filterWildcards, groupWildcardsByPrefix } from '@features/generation/core/wildcardCatalog';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { PANEL_HEADER_CONTROL_HEIGHT, PromptPanelHeader } from '@features/generation/ui/promptFields/PromptPanelHeader';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import { WildcardTransferActions } from '@features/generation/ui/promptFields/WildcardTransferActions';
import { WildcardDeleteError } from '@features/generation/ui/useWildcards';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { getListContentHeight, List } from '@platform/ui/list/List';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListSelectionBar } from '@platform/ui/list/ListSelectionBar';
import { LoadError } from '@platform/ui/LoadError';
import { Tooltip } from '@platform/ui/Tooltip';
import { CheckIcon, PencilIcon, PlusIcon, TrashIcon, XIcon } from 'lucide-react';
import { memo, useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** One value per line: the same shape the user is editing a variant in. */
const toValuesText = (values: string[]): string => values.join('\n');
// Normalize ingress so export and reimport preserve values.
const fromValuesText = (text: string): string[] => normalizeWildcardValues(text.split('\n'));

const VALUES_ERROR_KEY = {
  tooManyValues: 'widgets.generate.dynamicPrompts.wildcardTooManyValues',
  valueTooLong: 'widgets.generate.dynamicPrompts.wildcardValueTooLong',
} as const;

const NAME_ERROR_KEY = {
  invalid: 'widgets.generate.dynamicPrompts.wildcardNameInvalid',
  taken: 'widgets.generate.dynamicPrompts.wildcardNameTaken',
  tooLong: 'widgets.generate.dynamicPrompts.wildcardNameTooLong',
} as const;

const NO_SELECTION: ReadonlySet<string> = new Set();

/** A title over one description line, no media: the list's regular pitch. */
const LIST_DENSITY = 'regular';
/** Past this the list scrolls. */
const LIST_MAX_HEIGHT = '14rem';

/** Ungrouped names first and unheaded, then each prefix under its header, as one row sequence for the List. */
const toRows = (groups: ReturnType<typeof groupWildcardsByPrefix<WildcardRecord>>): ListRow<WildcardRecord>[] =>
  groups.flatMap((group) => [
    ...(group.label === null
      ? []
      : [{ count: group.wildcards.length, key: `header:${group.label}`, kind: 'header' as const, label: group.label }]),
    ...group.wildcards.map((wildcard) => ({ item: wildcard, key: wildcard.id, kind: 'item' as const })),
  ]);

interface WildcardDraft {
  id: string | null;
  name: string;
  valuesText: string;
}

export const WildcardsPanel = ({
  catalog,
  onInsert,
  showSyntaxHighlighting,
}: {
  catalog: WildcardCatalog;
  showSyntaxHighlighting: boolean;
  /** Splices `__name__` into the prompt at the caret. */
  onInsert: (reference: string) => void;
}) => {
  const { t } = useTranslation();
  const { notifications } = useGenerationUi();
  const nameFieldId = useId();
  const [draft, setDraft] = useState<WildcardDraft | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<WildcardRecord | null>(null);
  const [isBulkDeleteOpen, setIsBulkDeleteOpen] = useState(false);
  const [selectedIds, setSelectedIds] = useState<ReadonlySet<string>>(NO_SELECTION);
  const [searchTerm, setSearchTerm] = useState('');
  const selectionBarRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const newWildcardRef = useRef<HTMLButtonElement>(null);
  const filtered = useMemo(() => filterWildcards(catalog.wildcards, searchTerm), [catalog.wildcards, searchTerm]);
  const rows = useMemo(() => toRows(groupWildcardsByPrefix(filtered)), [filtered]);
  // Sized to the whole catalog, not the matches, so typing a search never resizes the popover under the caret.
  const listHeightPx = useMemo(
    () => getListContentHeight(toRows(groupWildcardsByPrefix(catalog.wildcards)), LIST_DENSITY),
    [catalog.wildcards]
  );
  // Ids the catalog no longer returns (deleted here, elsewhere, or by another tab) drop out of the selection.
  const selection = useMemo(() => {
    const liveIds = new Set(catalog.wildcards.map((wildcard) => wildcard.id));

    return [...selectedIds].every((id) => liveIds.has(id))
      ? selectedIds
      : new Set([...selectedIds].filter((id) => liveIds.has(id)));
  }, [catalog.wildcards, selectedIds]);
  const hasSelection = selection.size > 0;
  const hasUnselectedFiltered = useMemo(
    () => filtered.some((wildcard) => !selection.has(wildcard.id)),
    [filtered, selection]
  );
  // Exclude the current wildcard from rename collision checks.
  const nameError = draft
    ? getWildcardNameError(draft.name, new Set(catalog.wildcards.filter((w) => w.id !== draft.id).map((w) => w.name)))
    : null;
  // Keep editor and import bounds consistent.
  const valuesError = draft ? getWildcardValuesError(fromValuesText(draft.valuesText)) : null;
  // Prefer the server explanation over local bound errors.
  const draftError = error ?? (valuesError ? t(VALUES_ERROR_KEY[valuesError]) : null);

  const startCreate = useCallback(() => {
    setError(null);
    setDraft({ id: null, name: '', valuesText: '' });
  }, []);

  const startEdit = useCallback((wildcard: WildcardRecord) => {
    setError(null);
    setDraft({ id: wildcard.id, name: wildcard.name, valuesText: toValuesText(wildcard.values) });
  }, []);

  const cancel = useCallback(() => {
    setError(null);
    setDraft(null);
  }, []);

  const save = useCallback(async () => {
    if (!draft) {
      return;
    }

    const values = fromValuesText(draft.valuesText);

    try {
      if (draft.id === null) {
        await catalog.create({ name: draft.name, values });
      } else {
        await catalog.update(draft.id, { name: draft.name, values });
      }
      setDraft(null);
      setError(null);
    } catch (caught) {
      // Unwrap raw ApiError bodies before display.
      setError(getApiErrorMessage(caught, t('widgets.generate.dynamicPrompts.couldNotSaveWildcard')));
    }
  }, [catalog, draft, t]);

  const handleSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setSearchTerm(event.currentTarget.value),
    []
  );

  const closeDeleteDialog = useCallback(() => setPendingDelete(null), []);
  const closeBulkDeleteDialog = useCallback(() => setIsBulkDeleteOpen(false), []);

  const toggleSelected = useCallback((id: string) => {
    setSelectedIds((previous) => {
      const next = new Set(previous);

      if (!next.delete(id)) {
        next.add(id);
      }

      return next;
    });
  }, []);

  const handleToggleSelectAll = useCallback(() => {
    if (hasSelection && !hasUnselectedFiltered) {
      setSelectedIds(NO_SELECTION);

      return;
    }

    // Union: selections made under an earlier search survive, so the delete confirm counts every selected wildcard.
    setSelectedIds(new Set([...selection, ...filtered.map((wildcard) => wildcard.id)]));
  }, [filtered, hasSelection, hasUnselectedFiltered, selection]);

  /**
   * Where focus goes once the selection's own controls are gone: select all, or, with no rows left to select, the
   * search field, then New wildcard. Never the page body.
   */
  const getSelectionFocusTarget = useCallback((): HTMLElement | null => {
    const selectAll = selectionBarRef.current?.querySelector<HTMLInputElement>('input[type="checkbox"]');

    return selectAll && !selectAll.disabled ? selectAll : (searchInputRef.current ?? newWildcardRef.current);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedIds(NO_SELECTION);
    // The Clear selection button leaves with the selection; keep focus in the panel. With nothing listed, select all
    // turns disabled once the selection is gone, so the search field takes focus instead.
    (filtered.length === 0 ? (searchInputRef.current ?? newWildcardRef.current) : getSelectionFocusTarget())?.focus();
  }, [filtered.length, getSelectionFocusTarget]);

  /** Keeps what was not deleted selected, so a retry acts on exactly that. */
  const confirmBulkDelete = useCallback(async () => {
    const ids = [...selection];
    const keepUndeleted = (deletedIds: readonly string[]) => {
      const deleted = new Set(deletedIds);

      setSelectedIds(new Set(ids.filter((id) => !deleted.has(id))));
    };

    try {
      keepUndeleted(await catalog.removeMany(ids));
    } catch (caught) {
      const deletedIds = caught instanceof WildcardDeleteError ? caught.deletedIds : [];

      keepUndeleted(deletedIds);
      notifications.reportError({
        area: 'delete-wildcard',
        message: t('widgets.generate.dynamicPrompts.couldNotDeleteWildcards', {
          count: ids.length,
          deleted: deletedIds.length,
        }),
        namespace: 'generation',
      });
    }
  }, [catalog, notifications, selection, t]);

  /** Notify on failure because the confirmation dialog closes. */
  const confirmDelete = useCallback(async () => {
    if (!pendingDelete) {
      return;
    }

    try {
      await catalog.remove(pendingDelete.id);
    } catch (caught) {
      notifications.reportError({
        area: 'delete-wildcard',
        message: getApiErrorMessage(caught, t('widgets.generate.dynamicPrompts.couldNotDeleteWildcard')),
        namespace: 'generation',
      });
    }
  }, [catalog, notifications, pendingDelete, t]);

  // Read failed with nothing to show: the list says so, and controls that act on it go.
  const isUnreadable = catalog.isError && catalog.wildcards.length === 0;

  if (draft) {
    return (
      <Stack gap="2.5">
        <PromptPanelHeader
          label={
            draft.id === null
              ? t('widgets.generate.dynamicPrompts.newWildcard')
              : t('widgets.generate.dynamicPrompts.editWildcard')
          }
        />
        {/* Empty initial names disable save without premature validation errors. */}
        <Field
          error={nameError === null || nameError === 'empty' ? null : t(NAME_ERROR_KEY[nameError])}
          id={nameFieldId}
          label={t('widgets.generate.dynamicPrompts.wildcardName')}
        >
          <Input
            aria-invalid={nameError !== null && nameError !== 'empty' ? true : undefined}
            placeholder={t('widgets.generate.dynamicPrompts.wildcardNamePlaceholder')}
            value={draft.name}
            onChange={(event: ChangeEvent<HTMLInputElement>) => setDraft({ ...draft, name: event.currentTarget.value })}
          />
        </Field>
        {/* Wildcard values may contain nested dynamic syntax. */}
        <PromptTextarea
          aria-label={t('widgets.generate.dynamicPrompts.wildcardValues')}
          defaultHeightPx={144}
          fontSize="0.72rem"
          highlightDynamicPrompts
          knownWildcards={catalog.knownNames}
          maxHeightPx={320}
          minHeightPx={96}
          placeholder={t('widgets.generate.dynamicPrompts.wildcardValuesPlaceholder')}
          resizeHandleAriaLabel={t('widgets.generate.dynamicPrompts.resizeWildcardValues')}
          showLineNumbers
          showSyntaxHighlighting={showSyntaxHighlighting}
          value={draft.valuesText}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
            setDraft({ ...draft, valuesText: event.currentTarget.value })
          }
        />
        {draftError ? (
          <Text color="fg.error" fontSize="xs" wordBreak="break-word">
            {draftError}
          </Text>
        ) : null}
        <HStack justify="end">
          <Button variant="ghost" onClick={cancel}>
            <XIcon />
            {t('common.cancel')}
          </Button>
          <Button disabled={nameError !== null || valuesError !== null} onClick={() => void save()}>
            <CheckIcon />
            {t('common.save')}
          </Button>
        </HStack>
      </Stack>
    );
  }

  return (
    <Stack gap="2.5">
      <PromptPanelHeader label={t('widgets.generate.dynamicPrompts.wildcards')}>
        <Button ref={newWildcardRef} h={PANEL_HEADER_CONTROL_HEIGHT} size="sm" variant="ghost" onClick={startCreate}>
          <PlusIcon />
          {t('widgets.generate.dynamicPrompts.newWildcard')}
        </Button>
      </PromptPanelHeader>

      {/* A failed refresh keeps the wildcards read before; only a failure with nothing to show fills the list. */}
      {catalog.isError && catalog.wildcards.length > 0 ? (
        <LoadError
          detail={catalog.errorDetail}
          isRetrying={catalog.isRetrying}
          layout="inline"
          message={t('widgets.generate.dynamicPrompts.couldNotRefreshWildcards')}
          onRetry={catalog.retry}
        />
      ) : null}

      {/* Search, bar, rows and transfer actions read as one block, spaced at the row rhythm rather than the panel's. */}
      <Stack gap="1.5">
        {catalog.wildcards.length > 0 ? (
          <>
            {/* Its own field: the panel opens from the prompt's label, whose Field would otherwise lend this input its id,
              description and invalid state. */}
            <ChakraField.Root>
              <Input
                ref={searchInputRef}
                aria-label={t('widgets.generate.dynamicPrompts.searchWildcards')}
                placeholder={t('widgets.generate.dynamicPrompts.searchWildcards')}
                value={searchTerm}
                onChange={handleSearchChange}
              />
            </ChakraField.Root>
          </>
        ) : null}

        {/* The last row's slot already ends in a row gap, so one step less keeps the transfer row 1.5 off it too. */}
        <Stack gap="1">
          {/* The bar heads the list it selects in: it sits off the first row by the rows' own gap. */}
          <Stack gap="0.5">
            {/* Nothing to select while the list could not be read. */}
            {isUnreadable ? null : (
              <Box ref={selectionBarRef}>
                <ListSelectionBar
                  // No visible matches means indeterminate, not all selected; clicking clears the hidden selection.
                  checked={
                    hasSelection ? (hasUnselectedFiltered || filtered.length === 0 ? 'indeterminate' : true) : false
                  }
                  isDisabled={filtered.length === 0 && !hasSelection}
                  label={t('widgets.generate.dynamicPrompts.selectAllWildcards')}
                  summary={
                    hasSelection
                      ? t('widgets.generate.dynamicPrompts.selectedWildcardCount', { count: selection.size })
                      : undefined
                  }
                  onCheckedChange={handleToggleSelectAll}
                >
                  {hasSelection ? (
                    <>
                      <Button colorPalette="red" size="sm" variant="ghost" onClick={() => setIsBulkDeleteOpen(true)}>
                        <TrashIcon />
                        {t('common.delete')}
                      </Button>
                      <Tooltip content={t('widgets.generate.dynamicPrompts.clearWildcardSelection')}>
                        <IconButton
                          aria-label={t('widgets.generate.dynamicPrompts.clearWildcardSelection')}
                          size="sm"
                          variant="ghost"
                          onClick={clearSelection}
                        >
                          <XIcon />
                        </IconButton>
                      </Tooltip>
                    </>
                  ) : null}
                </ListSelectionBar>
              </Box>
            )}

            {/* Fits its rows up to the cap; empty, loading and error states take their own height. */}
            <Box h={listHeightPx > 0 ? `min(${LIST_MAX_HEIGHT}, ${listHeightPx}px)` : undefined}>
              <List
                density={LIST_DENSITY}
                emptyState={
                  catalog.isError && catalog.wildcards.length === 0 ? (
                    <LoadError
                      detail={catalog.errorDetail}
                      isRetrying={catalog.isRetrying}
                      layout="state"
                      message={t('widgets.generate.dynamicPrompts.couldNotLoadWildcards')}
                      onRetry={catalog.retry}
                    />
                  ) : (
                    <Text color="fg.subtle" fontSize="xs">
                      {catalog.isLoading
                        ? t('widgets.generate.dynamicPrompts.loadingWildcards')
                        : catalog.wildcards.length === 0
                          ? t('widgets.generate.dynamicPrompts.noWildcardsYet')
                          : t('widgets.generate.dynamicPrompts.noMatchingWildcards')}
                    </Text>
                  )
                }
                label={t('widgets.generate.dynamicPrompts.wildcards')}
                renderItem={(wildcard: WildcardRecord, rowProps: ListRowProps) => (
                  // Show full wildcard paths because rows display exact insertion syntax.
                  <WildcardRow
                    {...rowProps}
                    isSelected={selection.has(wildcard.id)}
                    wildcard={wildcard}
                    onDelete={setPendingDelete}
                    onEdit={startEdit}
                    onInsert={onInsert}
                    onToggleSelected={toggleSelected}
                  />
                )}
                rows={rows}
              />
            </Box>
          </Stack>

          {isUnreadable ? null : <WildcardTransferActions catalog={catalog} />}
        </Stack>
      </Stack>

      {/* Confirm irreversible deletion of handwritten values. */}
      <ConfirmDialog
        body={t('widgets.generate.dynamicPrompts.deleteWildcardBody', { name: pendingDelete?.name ?? '' })}
        confirmLabel={t('common.delete')}
        isOpen={pendingDelete !== null}
        title={t('widgets.generate.dynamicPrompts.deleteWildcardTitle')}
        onClose={closeDeleteDialog}
        onConfirm={confirmDelete}
      />
      <ConfirmDialog
        body={t('widgets.generate.dynamicPrompts.bulkDeleteWildcardsBody', { count: selection.size })}
        confirmLabel={t('widgets.generate.dynamicPrompts.bulkDeleteWildcardsConfirm', { count: selection.size })}
        isOpen={isBulkDeleteOpen}
        finalFocusEl={getSelectionFocusTarget}
        title={t('widgets.generate.dynamicPrompts.deleteSelectedWildcardsTitle')}
        onClose={closeBulkDeleteDialog}
        onConfirm={confirmBulkDelete}
      />
    </Stack>
  );
};

interface WildcardRowProps extends ListRowProps {
  wildcard: WildcardRecord;
  isSelected: boolean;
  onDelete: (wildcard: WildcardRecord) => void;
  onEdit: (wildcard: WildcardRecord) => void;
  onInsert: (reference: string) => void;
  onToggleSelected: (id: string) => void;
}

/** Memoized on its record and selection, so selecting all re-renders only the rows whose check changed. */
const WildcardRow = memo(function WildcardRow({
  isSelected,
  onDelete,
  onEdit,
  onInsert,
  onToggleSelected,
  wildcard,
  ...rowProps
}: WildcardRowProps) {
  const { t } = useTranslation();
  const reference = `__${wildcard.name}__`;

  return (
    <ListItem
      {...rowProps}
      actions={
        <>
          <Tooltip content={t('common.edit')}>
            <IconButton aria-label={t('common.edit')} size="sm" variant="ghost" onClick={() => onEdit(wildcard)}>
              <PencilIcon />
            </IconButton>
          </Tooltip>
          <Tooltip content={t('common.delete')}>
            <IconButton
              aria-label={t('common.delete')}
              colorPalette="red"
              size="sm"
              variant="ghost"
              onClick={() => onDelete(wildcard)}
            >
              <TrashIcon />
            </IconButton>
          </Tooltip>
        </>
      }
      description={
        wildcard.values.length > 0
          ? wildcard.values.join(', ')
          : t('widgets.generate.dynamicPrompts.wildcardHasNoValues')
      }
      isChecked={isSelected}
      title={reference}
      onCheckedChange={() => onToggleSelected(wildcard.id)}
      pressLabel={`${t('widgets.generate.dynamicPrompts.insertWildcard')}: ${reference}`}
      onPress={() => onInsert(reference)}
    />
  );
});
