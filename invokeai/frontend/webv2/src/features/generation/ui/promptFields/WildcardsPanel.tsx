/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { WildcardRecord } from '@features/generation/data/wildcards';
import type { WildcardCatalog } from '@features/generation/ui/useWildcards';
import type { ChangeEvent } from 'react';

import { HStack, Input, Separator, Stack, Text } from '@chakra-ui/react';
import { getWildcardNameError } from '@features/generation/core/dynamicPrompts';
import { filterWildcards, groupWildcardsByPrefix } from '@features/generation/core/wildcardCatalog';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { PANEL_HEADER_CONTROL_HEIGHT, PromptPanelHeader } from '@features/generation/ui/promptFields/PromptPanelHeader';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { CheckIcon, PencilIcon, PlusIcon, TrashIcon, XIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** One value per line: the same shape the user is editing a variant in. */
const toValuesText = (values: string[]): string => values.join('\n');
const fromValuesText = (text: string): string[] => text.split('\n');

const NAME_ERROR_KEY = {
  invalid: 'widgets.generate.dynamicPrompts.wildcardNameInvalid',
  taken: 'widgets.generate.dynamicPrompts.wildcardNameTaken',
  tooLong: 'widgets.generate.dynamicPrompts.wildcardNameTooLong',
} as const;

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
  const [searchTerm, setSearchTerm] = useState('');
  const groups = useMemo(
    () => groupWildcardsByPrefix(filterWildcards(catalog.wildcards, searchTerm)),
    [catalog.wildcards, searchTerm]
  );
  // Renaming to your own current name is not a clash, so the wildcard being
  // edited is excluded from the taken-names set.
  const nameError = draft
    ? getWildcardNameError(draft.name, new Set(catalog.wildcards.filter((w) => w.id !== draft.id).map((w) => w.name)))
    : null;

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
      // `ApiError.message` is the raw response body, so the backend's own
      // explanation only reads properly once it is unwrapped.
      setError(getApiErrorMessage(caught, t('widgets.generate.dynamicPrompts.couldNotSaveWildcard')));
    }
  }, [catalog, draft, t]);

  const handleSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setSearchTerm(event.currentTarget.value),
    []
  );

  const closeDeleteDialog = useCallback(() => setPendingDelete(null), []);

  /**
   * The dialog closes whether or not this throws, so a failure has nowhere inline
   * to land — it goes to the notification centre like the other Generation errors.
   */
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

  if (draft) {
    return (
      <Stack gap="2">
        <PromptPanelHeader
          label={
            draft.id === null
              ? t('widgets.generate.dynamicPrompts.newWildcard')
              : t('widgets.generate.dynamicPrompts.editWildcard')
          }
        />
        {/* An empty name is the starting state rather than a mistake, so it
            disables Save without being called out. */}
        <Field
          error={nameError === null || nameError === 'empty' ? null : t(NAME_ERROR_KEY[nameError])}
          id={nameFieldId}
          label={t('widgets.generate.dynamicPrompts.wildcardName')}
        >
          <Input
            aria-invalid={nameError !== null && nameError !== 'empty' ? true : undefined}
            placeholder={t('widgets.generate.dynamicPrompts.wildcardNamePlaceholder')}
            size="xs"
            value={draft.name}
            onChange={(event: ChangeEvent<HTMLInputElement>) => setDraft({ ...draft, name: event.currentTarget.value })}
          />
        </Field>
        {/* Values are expanded by dynamicprompts too, so a nested `{a|b}` or
            `__other__` is live syntax here and is coloured as such. The gutter
            numbers the values, which is what a line means in this editor. */}
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
          size="xs"
          value={draft.valuesText}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
            setDraft({ ...draft, valuesText: event.currentTarget.value })
          }
        />
        {error ? (
          <Text color="fg.error" fontSize="2xs" wordBreak="break-word">
            {error}
          </Text>
        ) : null}
        <HStack justify="end">
          <Button size="xs" variant="ghost" onClick={cancel}>
            <XIcon />
            {t('common.cancel')}
          </Button>
          <Button disabled={nameError !== null} size="xs" onClick={() => void save()}>
            <CheckIcon />
            {t('common.save')}
          </Button>
        </HStack>
      </Stack>
    );
  }

  return (
    <Stack gap="2">
      <PromptPanelHeader label={t('widgets.generate.dynamicPrompts.wildcards')}>
        <Button h={PANEL_HEADER_CONTROL_HEIGHT} size="2xs" variant="ghost" onClick={startCreate}>
          <PlusIcon />
          {t('widgets.generate.dynamicPrompts.newWildcard')}
        </Button>
      </PromptPanelHeader>

      {/* Searching an empty catalog would only offer a way to find nothing, so
          the input appears once there is something to search. */}
      {catalog.wildcards.length > 0 ? (
        <>
          <Input
            aria-label={t('widgets.generate.dynamicPrompts.searchWildcards')}
            placeholder={t('widgets.generate.dynamicPrompts.searchWildcards')}
            size="xs"
            value={searchTerm}
            onChange={handleSearchChange}
          />
          <Separator />
        </>
      ) : null}

      <Scrollable h="14rem" label={t('widgets.generate.dynamicPrompts.wildcards')}>
        {groups.length === 0 ? (
          <Text color="fg.subtle" fontSize="2xs" px="2" py="1.5">
            {catalog.wildcards.length === 0
              ? t('widgets.generate.dynamicPrompts.noWildcardsYet')
              : t('widgets.generate.dynamicPrompts.noMatchingWildcards')}
          </Text>
        ) : (
          <Stack gap="2">
            {groups.map((group) => (
              <Stack gap="0" key={group.label ?? ''}>
                {group.label === null ? null : (
                  <Text color="fg.subtle" fontSize="2xs" fontWeight="700" px="2" textTransform="uppercase">
                    {group.label}
                  </Text>
                )}
                {group.wildcards.map((wildcard) => (
                  // The row shows the whole `__animals/dogs__`, header or not:
                  // it is the exact text you would type, so trimming the prefix
                  // to match the header above would make it a lie.
                  <WildcardRow
                    key={wildcard.id}
                    wildcard={wildcard}
                    onDelete={setPendingDelete}
                    onEdit={startEdit}
                    onInsert={onInsert}
                  />
                ))}
              </Stack>
            ))}
          </Stack>
        )}
      </Scrollable>

      {/* A wildcard's values are typed by hand and there is no undo, so deleting
          one asks first — as every other destructive action here does. */}
      <ConfirmDialog
        body={t('widgets.generate.dynamicPrompts.deleteWildcardBody', { name: pendingDelete?.name ?? '' })}
        confirmLabel={t('common.delete')}
        isOpen={pendingDelete !== null}
        title={t('widgets.generate.dynamicPrompts.deleteWildcardTitle')}
        onClose={closeDeleteDialog}
        onConfirm={confirmDelete}
      />
    </Stack>
  );
};

const WildcardRow = ({
  onDelete,
  onEdit,
  onInsert,
  wildcard,
}: {
  wildcard: WildcardRecord;
  onDelete: (wildcard: WildcardRecord) => void;
  onEdit: (wildcard: WildcardRecord) => void;
  onInsert: (reference: string) => void;
}) => {
  const { t } = useTranslation();

  return (
    <HStack align="start" gap="1" pr="1">
      <Button
        alignItems="start"
        flex="1"
        h="auto"
        justifyContent="start"
        minW="0"
        px="2"
        py="1.5"
        size="xs"
        title={t('widgets.generate.dynamicPrompts.insertWildcard')}
        variant="ghost"
        onClick={() => onInsert(`__${wildcard.name}__`)}
      >
        <Stack align="start" gap="0" minW="0">
          <Text as="span" color="fg" fontFamily="mono" fontSize="0.72rem">
            __{wildcard.name}__
          </Text>
          <Text as="span" color="fg.subtle" fontSize="2xs" truncate>
            {wildcard.values.length > 0
              ? wildcard.values.join(', ')
              : t('widgets.generate.dynamicPrompts.wildcardHasNoValues')}
          </Text>
        </Stack>
      </Button>
      <Tooltip content={t('common.edit')}>
        <IconButton aria-label={t('common.edit')} size="2xs" variant="ghost" onClick={() => onEdit(wildcard)}>
          <PencilIcon />
        </IconButton>
      </Tooltip>
      <Tooltip content={t('common.delete')}>
        <IconButton
          aria-label={t('common.delete')}
          colorPalette="red"
          size="2xs"
          variant="ghost"
          onClick={() => onDelete(wildcard)}
        >
          <TrashIcon />
        </IconButton>
      </Tooltip>
    </HStack>
  );
};
