import type { SystemPromptRecord } from '@features/generation/data/systemPrompts';
import type { SystemPromptCatalog } from '@features/generation/ui/promptFields/useSystemPrompts';
import type { ChangeEvent } from 'react';

import { createListCollection, HStack, Input, Stack, Text, Textarea } from '@chakra-ui/react';
import { PANEL_HEADER_CONTROL_HEIGHT, PromptPanelHeader } from '@features/generation/ui/promptFields/PromptPanelHeader';
// Imported by subpath rather than from the `@platform/ui` barrel, which is at its
// direct-importer budget (a dev-invalidation limit) — depending on the components
// actually used means this module only rebuilds when those change.
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Scrollable } from '@platform/ui/Scrollable';
import { Select } from '@platform/ui/Select';
import { Tooltip } from '@platform/ui/Tooltip';
import { PencilIcon, PlusIcon, SettingsIcon, TrashIcon } from 'lucide-react';
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
}

const EMPTY_DRAFT: Draft = { content: '', name: '' };

const SystemPromptRow = ({
  canEdit,
  onDelete,
  onEdit,
  prompt,
}: {
  canEdit: boolean;
  prompt: SystemPromptRecord;
  onDelete: (prompt: SystemPromptRecord) => void;
  onEdit: (prompt: SystemPromptRecord) => void;
}) => {
  const { t } = useTranslation();
  const handleEdit = useCallback(() => onEdit(prompt), [onEdit, prompt]);
  const handleDelete = useCallback(() => onDelete(prompt), [onDelete, prompt]);

  return (
    <HStack justify="space-between" px="1" py="0.5">
      <MiddleTruncate fontSize="xs" minW="0" text={prompt.name} />
      {canEdit ? (
        <HStack gap="0.5">
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
        </HStack>
      ) : (
        // Shared by someone else. Says why it has no controls without adding colour.
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.generate.systemPrompts.shared')}
        </Text>
      )}
    </HStack>
  );
};

/**
 * The system-prompt control inside the Expand Prompt popover: a picker over the prompts this
 * account can see, and — behind the manage toggle — the list that creates, edits and deletes
 * them. Both live in one component because the popover shows exactly one of them at a time.
 */
export const SystemPromptsField = ({ catalog, onSelect, selectedId }: SystemPromptsFieldProps) => {
  const { canEdit, prompts } = catalog;
  const { t } = useTranslation();
  const selectId = useId();
  const nameFieldId = useId();
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
    setDraft({ content: record.content, name: record.name });
    setEditorTarget({ record });
  }, []);

  const closeEditor = useCallback(() => setEditorTarget(null), []);

  const saveDraft = useCallback(async () => {
    if (!editorTarget || !draft.name.trim() || !draft.content.trim()) {
      return;
    }

    setIsSaving(true);
    setError(null);

    let saved: SystemPromptRecord;

    try {
      saved = editorTarget.record ? await catalog.update(editorTarget.record, draft) : await catalog.create(draft);
    } catch (caught) {
      // `ApiError.message` is the raw response body, so the backend's own explanation only
      // reads properly once unwrapped. Reported in place — the popover has no toast surface.
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

    // The selection resolves against the visible list on read, so a deleted prompt needs no
    // corrective write here — but clearing it avoids one render with a dangling id.
    if (selectedId === pendingDelete.id) {
      onSelect(null);
    }
  }, [catalog, onSelect, pendingDelete, selectedId, t]);

  const cancelDelete = useCallback(() => setPendingDelete(null), []);
  const handleSave = useCallback(() => void saveDraft(), [saveDraft]);
  const handleConfirmDelete = useCallback(() => void confirmDelete(), [confirmDelete]);
  const handleNameChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setDraft((current) => ({ ...current, name: event.target.value })),
    []
  );
  const handleContentChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setDraft((current) => ({ ...current, content: event.target.value })),
    []
  );
  const handleSelectChange = useCallback(({ value }: { value: string[] }) => onSelect(value[0] ?? null), [onSelect]);

  const selectValue = useMemo(() => (selectedId ? [selectedId] : []), [selectedId]);

  if (isManaging && editorTarget) {
    return (
      <Stack gap="2">
        <PromptPanelHeader
          label={
            editorTarget.record ? t('widgets.generate.systemPrompts.edit') : t('widgets.generate.systemPrompts.new')
          }
        />
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
            disabled={!draft.name.trim() || !draft.content.trim()}
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
          valueText={t('widgets.generate.systemPrompts.none')}
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
