import type { SystemPromptRecord } from '@features/generation/data/systemPrompts';
import type { SystemPromptCatalog } from '@features/generation/ui/promptFields/useSystemPrompts';
import type { ChangeEvent } from 'react';

import { createListCollection, HStack, Input, Stack, Text, Textarea } from '@chakra-ui/react';
import {
  parseMaxTokensInput,
  SYSTEM_PROMPT_MAX_TOKENS_DEFAULT,
  SYSTEM_PROMPT_MAX_TOKENS_MAX,
  SYSTEM_PROMPT_MAX_TOKENS_MIN,
} from '@features/generation/core/systemPrompts';
import { PANEL_HEADER_CONTROL_HEIGHT, PromptPanelHeader } from '@features/generation/ui/promptFields/PromptPanelHeader';
// Use subpaths to respect the importer budget.
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Scrollable } from '@platform/ui/Scrollable';
import { Select } from '@platform/ui/Select';
import { Tooltip } from '@platform/ui/Tooltip';
import { CopyIcon, PencilIcon, PlusIcon, SettingsIcon, TrashIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface SystemPromptsFieldProps {
  catalog: SystemPromptCatalog;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

/** `record: null` means the editor is composing a new prompt. */
type EditorTarget = { record: SystemPromptRecord | null };

interface Draft {
  name: string;
  content: string;
  /** Raw input text — see `parseMaxTokensInput`. Empty means "use the backend default". */
  maxTokens: string;
}

const EMPTY_DRAFT: Draft = { content: '', maxTokens: '', name: '' };

const SystemPromptRow = ({
  canEdit,
  onDelete,
  onDuplicate,
  onEdit,
  prompt,
}: {
  canEdit: boolean;
  prompt: SystemPromptRecord;
  onDelete: (prompt: SystemPromptRecord) => void;
  onDuplicate: (prompt: SystemPromptRecord) => void;
  onEdit: (prompt: SystemPromptRecord) => void;
}) => {
  const { t } = useTranslation();
  const handleEdit = useCallback(() => onEdit(prompt), [onEdit, prompt]);
  const handleDelete = useCallback(() => onDelete(prompt), [onDelete, prompt]);
  const handleDuplicate = useCallback(() => onDuplicate(prompt), [onDuplicate, prompt]);

  return (
    <HStack justify="space-between" px="1" py="0.5">
      <MiddleTruncate fontSize="xs" minW="0" text={prompt.name} />
      <HStack gap="0.5">
        {/* Copying remains available without edit rights over the source. */}
        <Tooltip content={t('widgets.generate.systemPrompts.duplicate')}>
          <IconButton
            aria-label={t('widgets.generate.systemPrompts.duplicate')}
            size="2xs"
            variant="ghost"
            onClick={handleDuplicate}
          >
            <CopyIcon />
          </IconButton>
        </Tooltip>
        {canEdit ? (
          <>
            <Tooltip content={t('common.edit')}>
              <IconButton aria-label={t('common.edit')} size="2xs" variant="ghost" onClick={handleEdit}>
                <PencilIcon />
              </IconButton>
            </Tooltip>
            <Tooltip content={t('common.delete')}>
              <IconButton aria-label={t('common.delete')} size="2xs" variant="ghost" onClick={handleDelete}>
                <TrashIcon />
              </IconButton>
            </Tooltip>
          </>
        ) : (
          // Shared by someone else. Says why it has no edit controls without adding colour.
          <Text color="fg.subtle" fontSize="2xs">
            {t('widgets.generate.systemPrompts.shared')}
          </Text>
        )}
      </HStack>
    </HStack>
  );
};

export const SystemPromptsField = ({ catalog, onSelect, selectedId }: SystemPromptsFieldProps) => {
  const { canEdit, prompts } = catalog;
  const { t } = useTranslation();
  const selectId = useId();
  const nameFieldId = useId();
  const maxTokensFieldId = useId();
  const [isManaging, setIsManaging] = useState(false);
  const [editorTarget, setEditorTarget] = useState<EditorTarget | null>(null);
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT);
  const [pendingDelete, setPendingDelete] = useState<SystemPromptRecord | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const collection = useMemo(
    () => createListCollection({ items: prompts.map(({ id, name }) => ({ label: name, value: id })) }),
    [prompts]
  );

  const toggleManaging = useCallback(() => {
    setIsManaging((managing) => !managing);
    // Reopening should land on the list, not resume a half-written prompt.
    setEditorTarget(null);
  }, []);

  const startCreate = useCallback(() => {
    setDraft(EMPTY_DRAFT);
    setEditorTarget({ record: null });
  }, []);

  const startEdit = useCallback((record: SystemPromptRecord) => {
    setDraft({
      content: record.content,
      maxTokens: record.maxTokens === null ? '' : String(record.maxTokens),
      name: record.name,
    });
    setEditorTarget({ record });
  }, []);

  const closeEditor = useCallback(() => setEditorTarget(null), []);

  const saveDraft = useCallback(async () => {
    if (!editorTarget || !draft.name.trim() || !draft.content.trim()) {
      return;
    }

    const maxTokens = parseMaxTokensInput(draft.maxTokens);

    // Validate on submit to cover races after the disabled-button render.
    if (maxTokens === 'invalid') {
      setError(
        t('widgets.generate.systemPrompts.maxTokensInvalid', {
          max: SYSTEM_PROMPT_MAX_TOKENS_MAX,
          min: SYSTEM_PROMPT_MAX_TOKENS_MIN,
        })
      );
      return;
    }

    const payload = { content: draft.content, maxTokens, name: draft.name };

    setIsSaving(true);
    setError(null);

    let saved: SystemPromptRecord;

    try {
      saved = editorTarget.record ? await catalog.update(editorTarget.record, payload) : await catalog.create(payload);
    } catch (caught) {
      // Unwrap errors inline; this popover has no toast surface.
      setError(getApiErrorMessage(caught, t('widgets.generate.systemPrompts.couldNotSave')));
      return;
    } finally {
      setIsSaving(false);
    }

    // A newly created prompt becomes the selection — it is almost always why it was written.
    if (!editorTarget.record) {
      onSelect(saved.id);
    }

    setEditorTarget(null);
  }, [catalog, draft, editorTarget, onSelect, t]);

  const duplicatePrompt = useCallback(
    async (prompt: SystemPromptRecord) => {
      setError(null);

      let copy: SystemPromptRecord;

      try {
        copy = await catalog.duplicate(prompt);
      } catch (caught) {
        setError(getApiErrorMessage(caught, t('widgets.generate.systemPrompts.couldNotSave')));
        return;
      }

      // A copy is made to be used or edited, so hand the user straight to it.
      onSelect(copy.id);
      startEdit(copy);
    },
    [catalog, onSelect, startEdit, t]
  );

  const confirmDelete = useCallback(async () => {
    if (!pendingDelete) {
      return;
    }

    setError(null);

    try {
      await catalog.remove(pendingDelete);
    } catch (caught) {
      setError(getApiErrorMessage(caught, t('widgets.generate.systemPrompts.couldNotDelete')));
      return;
    } finally {
      setPendingDelete(null);
    }

    // Clear selection to avoid a transient dangling ID even though reads have a fallback.
    if (selectedId === pendingDelete.id) {
      onSelect(null);
    }
  }, [catalog, onSelect, pendingDelete, selectedId, t]);

  const cancelDelete = useCallback(() => setPendingDelete(null), []);
  const handleSave = useCallback(() => void saveDraft(), [saveDraft]);
  const handleDuplicate = useCallback((prompt: SystemPromptRecord) => void duplicatePrompt(prompt), [duplicatePrompt]);
  const handleConfirmDelete = useCallback(() => void confirmDelete(), [confirmDelete]);
  const handleNameChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setDraft((current) => ({ ...current, name: event.target.value })),
    []
  );
  const handleContentChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setDraft((current) => ({ ...current, content: event.target.value })),
    []
  );
  const handleMaxTokensChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setDraft((current) => ({ ...current, maxTokens: event.target.value })),
    []
  );
  const handleSelectChange = useCallback(({ value }: { value: string[] }) => onSelect(value[0] ?? null), [onSelect]);

  const isMaxTokensInvalid = parseMaxTokensInput(draft.maxTokens) === 'invalid';
  const maxTokensError = isMaxTokensInvalid
    ? t('widgets.generate.systemPrompts.maxTokensInvalid', {
        max: SYSTEM_PROMPT_MAX_TOKENS_MAX,
        min: SYSTEM_PROMPT_MAX_TOKENS_MIN,
      })
    : null;

  const selectValue = useMemo(() => (selectedId ? [selectedId] : []), [selectedId]);
  // Keep valueText tied to selection because Select renders overrides verbatim.
  const selectedName = useMemo(
    () => (selectedId ? (prompts.find((prompt) => prompt.id === selectedId)?.name ?? null) : null),
    [prompts, selectedId]
  );

  if (isManaging && editorTarget) {
    return (
      <Stack gap="2">
        <PromptPanelHeader
          label={
            editorTarget.record ? t('widgets.generate.systemPrompts.edit') : t('widgets.generate.systemPrompts.new')
          }
        />
        {editorTarget.record?.isPublic ? (
          <Text color="fg.muted" fontSize="2xs">
            {t('widgets.generate.systemPrompts.sharedEditWarning')}
          </Text>
        ) : null}
        <Field id={nameFieldId} label={t('widgets.generate.systemPrompts.name')}>
          <Input
            id={nameFieldId}
            placeholder={t('widgets.generate.systemPrompts.namePlaceholder')}
            size="xs"
            value={draft.name}
            onChange={handleNameChange}
          />
        </Field>
        <Textarea
          aria-label={t('widgets.generate.systemPrompts.content')}
          minH="6rem"
          placeholder={t('widgets.generate.systemPrompts.contentPlaceholder')}
          size="xs"
          value={draft.content}
          onChange={handleContentChange}
        />
        <Field
          error={maxTokensError}
          helpText={t('widgets.generate.systemPrompts.maxTokensHelp', { tokens: SYSTEM_PROMPT_MAX_TOKENS_DEFAULT })}
          id={maxTokensFieldId}
          label={t('widgets.generate.systemPrompts.maxTokens')}
        >
          <Input
            id={maxTokensFieldId}
            // Use inputMode instead of number input so empty-means-default drafts survive.
            inputMode="numeric"
            placeholder={String(SYSTEM_PROMPT_MAX_TOKENS_DEFAULT)}
            size="xs"
            value={draft.maxTokens}
            onChange={handleMaxTokensChange}
          />
        </Field>
        {error ? (
          <Text color="fg.error" fontSize="xs">
            {error}
          </Text>
        ) : null}
        <HStack justify="flex-end">
          <Button size="xs" variant="ghost" onClick={closeEditor}>
            {t('common.cancel')}
          </Button>
          <Button
            disabled={!draft.name.trim() || !draft.content.trim() || isMaxTokensInvalid}
            loading={isSaving}
            size="xs"
            onClick={handleSave}
          >
            {t('common.save')}
          </Button>
        </HStack>
      </Stack>
    );
  }

  if (isManaging) {
    return (
      <Stack gap="2">
        <PromptPanelHeader label={t('widgets.generate.systemPrompts.title')}>
          <Button h={PANEL_HEADER_CONTROL_HEIGHT} size="2xs" variant="ghost" onClick={startCreate}>
            <PlusIcon />
            {t('widgets.generate.systemPrompts.new')}
          </Button>
        </PromptPanelHeader>
        {prompts.length === 0 ? (
          <Text color="fg.subtle" fontSize="xs">
            {t('widgets.generate.systemPrompts.none')}
          </Text>
        ) : (
          <Scrollable maxH="12rem">
            <Stack gap="0.5">
              {prompts.map((prompt) => (
                <SystemPromptRow
                  key={prompt.id}
                  canEdit={canEdit(prompt)}
                  prompt={prompt}
                  onDelete={setPendingDelete}
                  onDuplicate={handleDuplicate}
                  onEdit={startEdit}
                />
              ))}
            </Stack>
          </Scrollable>
        )}
        {error ? (
          <Text color="fg.error" fontSize="xs">
            {error}
          </Text>
        ) : null}
        <Button size="xs" variant="ghost" onClick={toggleManaging}>
          {t('common.done')}
        </Button>
        <ConfirmDialog
          body={t('widgets.generate.systemPrompts.deleteBody', { name: pendingDelete?.name ?? '' })}
          confirmLabel={t('common.delete')}
          isOpen={pendingDelete !== null}
          title={t('widgets.generate.systemPrompts.deleteTitle')}
          onClose={cancelDelete}
          onConfirm={handleConfirmDelete}
        />
      </Stack>
    );
  }

  return (
    <Field id={selectId} label={t('widgets.generate.systemPrompts.title')}>
      <HStack gap="1">
        <Select
          aria-label={t('widgets.generate.systemPrompts.title')}
          collection={collection}
          disabled={prompts.length === 0}
          flex="1"
          id={selectId}
          size="xs"
          value={selectValue}
          valueText={selectedName ?? t('widgets.generate.systemPrompts.none')}
          onValueChange={handleSelectChange}
        />
        <Tooltip content={t('widgets.generate.systemPrompts.manage')}>
          <IconButton
            aria-label={t('widgets.generate.systemPrompts.manage')}
            size="2xs"
            variant="ghost"
            onClick={toggleManaging}
          >
            <SettingsIcon />
          </IconButton>
        </Tooltip>
      </HStack>
    </Field>
  );
};
