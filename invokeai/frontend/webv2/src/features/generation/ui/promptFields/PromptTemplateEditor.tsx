/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type {
  PromptTemplateCreateDraft,
  PromptTemplateImageUpdate,
  PromptTemplateRecord,
  PromptTemplateUpdateDraft,
} from '@features/generation/data/promptTemplates';
import type { PendingPromptTemplateDraft } from '@features/generation/ui/promptTemplateDraftStore';
import type { PromptTemplateCatalog } from '@features/generation/ui/usePromptTemplates';
import type { ChangeEvent } from 'react';

import { HStack, Input, Stack, Text } from '@chakra-ui/react';
import { PROMPT_TEMPLATE_PLACEHOLDER } from '@features/generation/core/promptTemplates';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { PromptPanelHeader } from '@features/generation/ui/promptFields/PromptPanelHeader';
import { PromptTemplateImage } from '@features/generation/ui/promptFields/PromptTemplateImage';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { DropZone } from '@platform/ui/DropZone';
import { Field } from '@platform/ui/Field';
import { Tooltip } from '@platform/ui/Tooltip';
import { CheckIcon, ImageUpIcon, XIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
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
  image: PromptTemplateImageUpdate;
  /** Undefined keeps the stored image, a URL previews a replacement, null removes it. */
  imagePreviewUrl?: string | null;
}

const MAX_NAME_LENGTH = 128;
const NEW_TEMPLATE_IMAGE = { hasImage: false, id: 'new' } as const;

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
    image: { kind: 'preserve' },
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
   * ten images held on to all ten. Stored images are owned by the shared image
   * resource; only local replacement previews are tracked here.
   */
  const takeObjectUrl = useCallback((file: Blob | null): string | null => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }

    objectUrlRef.current = file ? URL.createObjectURL(file) : null;

    return objectUrlRef.current;
  }, []);

  useMountEffect(() => () => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }
  });

  const trimmedName = draft.name.trim();
  const isPlaceholderPresent = draft.positivePrompt.includes(PROMPT_TEMPLATE_PLACEHOLDER);
  const nameError = trimmedName.length > MAX_NAME_LENGTH ? t('widgets.generate.promptTemplates.nameTooLong') : null;
  const hasImage = draft.image.kind === 'replace' || (draft.image.kind === 'preserve' && template?.hasImage === true);

  const save = useCallback(async () => {
    setIsSaving(true);
    setError(null);

    let saved: PromptTemplateRecord;

    try {
      if (template) {
        const nextDraft: PromptTemplateUpdateDraft = {
          image: draft.image,
          name: trimmedName,
          negativePrompt: draft.negativePrompt,
          positivePrompt: draft.positivePrompt,
        };
        saved = await catalog.update(template, nextDraft);
      } else {
        const nextDraft: PromptTemplateCreateDraft = {
          image: draft.image.kind === 'replace' ? draft.image.blob : null,
          name: trimmedName,
          negativePrompt: draft.negativePrompt,
          positivePrompt: draft.positivePrompt,
        };
        saved = await catalog.create(nextDraft);
      }
    } catch (caught) {
      // `ApiError.message` is the raw response body, so the backend's own
      // explanation only reads properly once it is unwrapped.
      setError(getApiErrorMessage(caught, t('widgets.generate.promptTemplates.couldNotSave')));
      return;
    } finally {
      setIsSaving(false);
    }

    // Deliberately outside the `try`. Handing the saved record back is the
    // caller's business, and a failure there is not a failure to save — inside,
    // it was reported as "could not save this template" about a template that
    // had just been saved. It reaches the caller's own reporting instead.
    onSaved(saved);
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

        setDraft((current) => ({
          ...current,
          image: { blob: file, kind: 'replace' },
          imagePreviewUrl,
        }));
      }

      // Reset so re-picking the same file still fires a change.
      event.currentTarget.value = '';
    },
    [takeObjectUrl]
  );

  const clearImage = useCallback(() => {
    takeObjectUrl(null);
    setDraft((current) => ({ ...current, image: { kind: 'remove' }, imagePreviewUrl: null }));
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
          {template ? (
            <PromptTemplateImage
              alt=""
              boxSize="12"
              flexShrink="0"
              fallback={null}
              localPreviewUrl={draft.imagePreviewUrl}
              objectFit="cover"
              rounded="md"
              template={template}
            />
          ) : draft.imagePreviewUrl ? (
            <PromptTemplateImage
              alt=""
              boxSize="12"
              fallback={null}
              localPreviewUrl={draft.imagePreviewUrl}
              objectFit="cover"
              rounded="md"
              template={NEW_TEMPLATE_IMAGE}
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
              {hasImage
                ? t('widgets.generate.promptTemplates.replaceImage')
                : t('widgets.generate.promptTemplates.addImage')}
            </Text>
          </DropZone>
          {hasImage ? (
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
