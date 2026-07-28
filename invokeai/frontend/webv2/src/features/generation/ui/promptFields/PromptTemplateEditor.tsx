/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { PromptTemplateDraft, PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type { PendingPromptTemplateDraft } from '@features/generation/ui/promptTemplateDraftStore';
import type { PromptTemplateCatalog } from '@features/generation/ui/usePromptTemplates';
import type { ChangeEvent } from 'react';

import { HStack, Image, Input, Stack, Text } from '@chakra-ui/react';
import { PROMPT_TEMPLATE_PLACEHOLDER } from '@features/generation/core/promptTemplates';
import { fetchPromptTemplateImage } from '@features/generation/data/promptTemplates';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { PromptPanelHeader } from '@features/generation/ui/promptFields/PromptPanelHeader';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { DropZone } from '@platform/ui/DropZone';
import { Field } from '@platform/ui/Field';
import { Tooltip } from '@platform/ui/Tooltip';
import { CheckIcon, ImageUpIcon, XIcon } from 'lucide-react';
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface PromptTemplateEditorProps {
  catalog: PromptTemplateCatalog;
  /** The template being edited, or null when composing a new one. */
  template: PromptTemplateRecord | null;
  /** Prompts to start a new template from, e.g. handed over from an image. */
  prefill?: PendingPromptTemplateDraft;
  showSyntaxHighlighting: boolean;
  onCancel: () => void;
  onSaved: (template: PromptTemplateRecord) => void;
}

interface EditorDraft {
  name: string;
  negativePrompt: string;
  positivePrompt: string;
  image: Blob | null;
  imagePreviewUrl: string | null;
  /**
   * Whether the existing image has been dealt with — fetched, replaced, or
   * removed. Stated outright because `image: null` is both "not loaded yet" and
   * "the user took it off", and reading it as the former let a removal be undone
   * by a fetch that landed afterwards.
   */
  isExistingImageSettled: boolean;
}

const MAX_NAME_LENGTH = 128;

export const PromptTemplateEditor = ({
  catalog,
  onCancel,
  onSaved,
  prefill,
  showSyntaxHighlighting,
  template,
}: PromptTemplateEditorProps) => {
  const { t } = useTranslation();
  const { notifications } = useGenerationUi();
  const nameFieldId = useId();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [draft, setDraft] = useState<EditorDraft>({
    image: null,
    imagePreviewUrl: template?.imageUrl ?? null,
    isExistingImageSettled: !template?.imageUrl,
    name: template?.name ?? '',
    negativePrompt: template?.negativePrompt ?? prefill?.negativePrompt ?? '',
    positivePrompt: template?.positivePrompt ?? prefill?.positivePrompt ?? '',
  });
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  /**
   * Swaps in a preview URL for a picked file, releasing the previous one.
   *
   * A blob URL pins the whole file for the document's lifetime, so re-picking
   * ten images held on to all ten. The remote `template.imageUrl` shares this
   * field but is not ours to revoke, which is why only what we created is
   * tracked here.
   */
  const takeObjectUrl = useCallback((file: Blob | null): string | null => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }

    objectUrlRef.current = file ? URL.createObjectURL(file) : null;

    return objectUrlRef.current;
  }, []);

  useEffect(
    () => () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    },
    []
  );

  // The backend replaces the whole record, so saving without resending the
  // existing image would drop it. Load it up front and treat it as part of the
  // draft from then on.
  useMountEffect(() => {
    const imageUrl = template?.imageUrl;

    if (!imageUrl) {
      return;
    }

    void fetchPromptTemplateImage(imageUrl).then((image) => {
      if (image) {
        setDraft((current) =>
          current.isExistingImageSettled ? current : { ...current, image, isExistingImageSettled: true }
        );
      }
    });
  });

  const trimmedName = draft.name.trim();
  const isPlaceholderPresent = draft.positivePrompt.includes(PROMPT_TEMPLATE_PLACEHOLDER);
  const nameError = trimmedName.length > MAX_NAME_LENGTH ? t('widgets.generate.promptTemplates.nameTooLong') : null;

  const save = useCallback(async () => {
    const nextDraft: PromptTemplateDraft = {
      image: draft.image,
      name: trimmedName,
      negativePrompt: draft.negativePrompt,
      positivePrompt: draft.positivePrompt,
    };

    setIsSaving(true);
    setError(null);

    try {
      const saved = template ? await catalog.update(template.id, nextDraft) : await catalog.create(nextDraft);

      onSaved(saved);
    } catch (caught) {
      // `ApiError.message` is the raw response body, so the backend's own
      // explanation only reads properly once it is unwrapped.
      setError(getApiErrorMessage(caught, t('widgets.generate.promptTemplates.couldNotSave')));
    } finally {
      setIsSaving(false);
    }
  }, [catalog, draft, onSaved, t, template, trimmedName]);

  const insertPlaceholder = useCallback(
    () =>
      setDraft((current) => ({
        ...current,
        positivePrompt: current.positivePrompt
          ? `${current.positivePrompt} ${PROMPT_TEMPLATE_PLACEHOLDER}`
          : PROMPT_TEMPLATE_PLACEHOLDER,
      })),
    []
  );

  const pickImage = useCallback(() => fileInputRef.current?.click(), []);

  // The value has to be read before `setDraft`, not inside the updater: React
  // nulls `currentTarget` once the handler returns, and an updater can run after
  // that.
  const updateDraftField = useCallback(
    (field: 'name' | 'negativePrompt' | 'positivePrompt', value: string) =>
      setDraft((current) => ({ ...current, [field]: value })),
    []
  );

  const handleImageChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];

      if (file) {
        const imagePreviewUrl = takeObjectUrl(file);

        setDraft((current) => ({ ...current, image: file, imagePreviewUrl, isExistingImageSettled: true }));
      }

      // Reset so re-picking the same file still fires a change.
      event.currentTarget.value = '';
    },
    [takeObjectUrl]
  );

  const clearImage = useCallback(() => {
    takeObjectUrl(null);
    setDraft((current) => ({ ...current, image: null, imagePreviewUrl: null, isExistingImageSettled: true }));
  }, [takeObjectUrl]);

  const reportSaveError = useCallback(
    (caught: unknown) =>
      notifications.reportError({
        area: 'save-prompt-template',
        message: getApiErrorMessage(caught, t('widgets.generate.promptTemplates.couldNotSave')),
        namespace: 'generation',
      }),
    [notifications, t]
  );

  const handleSave = useCallback(() => void save().catch(reportSaveError), [reportSaveError, save]);

  const insertPlaceholderControl = useMemo(
    () => (
      <Tooltip
        content={
          isPlaceholderPresent
            ? t('widgets.generate.promptTemplates.placeholderAlreadyUsed')
            : t('widgets.generate.promptTemplates.insertPlaceholderHelp')
        }
      >
        <Button disabled={isPlaceholderPresent} size="2xs" variant="ghost" onClick={insertPlaceholder}>
          {PROMPT_TEMPLATE_PLACEHOLDER}
        </Button>
      </Tooltip>
    ),
    [insertPlaceholder, isPlaceholderPresent, t]
  );

  return (
    <Stack gap="2">
      <PromptPanelHeader
        label={
          template
            ? t('widgets.generate.promptTemplates.editTemplate')
            : t('widgets.generate.promptTemplates.newTemplate')
        }
      />

      <Field
        error={nameError}
        id={nameFieldId}
        label={t('widgets.generate.promptTemplates.name')}
        helpText={t('widgets.generate.promptTemplates.nameHelp')}
      >
        <Input
          aria-invalid={nameError !== null ? true : undefined}
          id={nameFieldId}
          placeholder={t('widgets.generate.promptTemplates.namePlaceholder')}
          size="xs"
          value={draft.name}
          onChange={(event: ChangeEvent<HTMLInputElement>) => updateDraftField('name', event.currentTarget.value)}
        />
      </Field>

      <Field label={t('common.prompt')} labelEnd={insertPlaceholderControl}>
        {/* `highlightDynamicPrompts` stays off here: `{prompt}` is a template
            placeholder, and colouring it as a dynamic-prompt group would promise
            an expansion that the merge consumes before one could happen. */}
        <PromptTextarea
          aria-label={t('widgets.generate.promptTemplates.positivePrompt')}
          defaultHeightPx={80}
          fontSize="0.72rem"
          maxHeightPx={240}
          minHeightPx={64}
          placeholder={t('widgets.generate.promptTemplates.positivePromptPlaceholder')}
          resizeHandleAriaLabel={t('widgets.generate.promptTemplates.resizePositivePrompt')}
          showSyntaxHighlighting={showSyntaxHighlighting}
          size="xs"
          value={draft.positivePrompt}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
            updateDraftField('positivePrompt', event.currentTarget.value)
          }
        />
      </Field>

      <Field label={t('common.negative')}>
        <PromptTextarea
          aria-label={t('widgets.generate.promptTemplates.negativePrompt')}
          defaultHeightPx={56}
          fontSize="0.72rem"
          maxHeightPx={200}
          minHeightPx={56}
          placeholder={t('widgets.generate.promptTemplates.negativePromptPlaceholder')}
          resizeHandleAriaLabel={t('widgets.generate.promptTemplates.resizeNegativePrompt')}
          showSyntaxHighlighting={showSyntaxHighlighting}
          size="xs"
          value={draft.negativePrompt}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
            updateDraftField('negativePrompt', event.currentTarget.value)
          }
        />
      </Field>

      <Field label={t('widgets.generate.promptTemplates.image')}>
        <HStack gap="2">
          {draft.imagePreviewUrl ? (
            <Image
              alt=""
              boxSize="12"
              borderColor="border.emphasized"
              borderWidth="1px"
              flexShrink="0"
              objectFit="cover"
              rounded="md"
              src={draft.imagePreviewUrl}
            />
          ) : null}
          <DropZone
            alignItems="center"
            as="button"
            cursor="pointer"
            display="flex"
            flex="1"
            gap="1.5"
            justifyContent="center"
            px="2"
            py="2.5"
            onClick={pickImage}
          >
            <ImageUpIcon size={14} />
            <Text as="span" fontSize="2xs">
              {draft.imagePreviewUrl
                ? t('widgets.generate.promptTemplates.replaceImage')
                : t('widgets.generate.promptTemplates.addImage')}
            </Text>
          </DropZone>
          {draft.imagePreviewUrl ? (
            <Tooltip content={t('widgets.generate.promptTemplates.removeImage')}>
              <IconButton
                aria-label={t('widgets.generate.promptTemplates.removeImage')}
                size="2xs"
                variant="ghost"
                onClick={clearImage}
              >
                <XIcon />
              </IconButton>
            </Tooltip>
          ) : null}
          <input accept="image/*" hidden ref={fileInputRef} type="file" onChange={handleImageChange} />
        </HStack>
      </Field>

      {error ? (
        <Text color="fg.error" fontSize="2xs" wordBreak="break-word">
          {error}
        </Text>
      ) : null}

      <HStack justify="end">
        <Button disabled={isSaving} size="xs" variant="ghost" onClick={onCancel}>
          <XIcon />
          {t('common.cancel')}
        </Button>
        <Button disabled={!trimmedName || nameError !== null} loading={isSaving} size="xs" onClick={handleSave}>
          <CheckIcon />
          {t('common.save')}
        </Button>
      </HStack>
    </Stack>
  );
};
