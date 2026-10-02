import type { GalleryItem } from '@features/gallery';
import type { GalleryMediaSlotLabels, GalleryMediaSlotValue } from '@features/gallery/mediaSlot';
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
import { GalleryMediaSlot } from '@features/gallery/mediaSlot';
import { PROMPT_TEMPLATE_PLACEHOLDER } from '@features/generation/core/promptTemplates';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { PromptPanelHeader } from '@features/generation/ui/promptFields/PromptPanelHeader';
import { PromptTemplateImage } from '@features/generation/ui/promptFields/PromptTemplateImage';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { Field } from '@platform/ui/Field';
import { Tooltip } from '@platform/ui/Tooltip';
import { CheckIcon, XIcon } from 'lucide-react';
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
  /** What the slot names the replacement: the picked item or file. */
  imageName?: string;
}

const MAX_NAME_LENGTH = 128;
const NEW_TEMPLATE_IMAGE = { hasImage: false, id: 'new' } as const;
const IMAGE_ONLY = ['image'] as const;
const IMAGE_DROP_ID = 'prompt-template-image';

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
  const [draft, setDraft] = useState<EditorDraft>({
    image: { kind: 'preserve' },
    name: template?.name ?? '',
    negativePrompt: template?.negativePrompt ?? prefill?.negativePrompt ?? '',
    positivePrompt: template?.positivePrompt ?? prefill?.positivePrompt ?? '',
  });
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const galleryFetchRef = useRef<AbortController | null>(null);

  /** Revoke replaced local blob URLs; shared stored-image resources own their own lifetime. */
  const takeObjectUrl = useCallback((file: Blob | null): string | null => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }

    objectUrlRef.current = file ? URL.createObjectURL(file) : null;

    return objectUrlRef.current;
  }, []);

  useMountEffect(() => () => {
    galleryFetchRef.current?.abort();
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
      // Unwrap raw ApiError bodies for display.
      setError(getApiErrorMessage(caught, t('widgets.generate.promptTemplates.couldNotSave')));
      return;
    } finally {
      setIsSaving(false);
    }

    // Run post-save callbacks outside save error handling so callback failure cannot misreport the save.
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

  // Read currentTarget before deferred state updates; React clears it after dispatch.
  const updateDraftField = useCallback(
    (field: 'name' | 'negativePrompt' | 'positivePrompt', value: string) =>
      setDraft((current) => ({ ...current, [field]: value })),
    []
  );

  /** Cancels a pending gallery fetch and hands back a controller for the next one. */
  const abortGalleryFetch = useCallback((): AbortController => {
    galleryFetchRef.current?.abort();
    galleryFetchRef.current = new AbortController();
    return galleryFetchRef.current;
  }, []);

  const setImageBlob = useCallback(
    (blob: Blob, imageName: string) => {
      const imagePreviewUrl = takeObjectUrl(blob);

      setDraft((current) => ({ ...current, image: { blob, kind: 'replace' }, imageName, imagePreviewUrl }));
    },
    [takeObjectUrl]
  );

  // Only the latest pick owns the result; abort on replacement, clear, or unmount.
  const handleGalleryPick = useCallback(
    (item: GalleryItem) => {
      const controller = abortGalleryFetch();
      const owner = captureAccountScope();
      const signal = AbortSignal.any([controller.signal, owner.signal]);

      void fetch(item.fullUrl, { signal })
        .then((response) => {
          if (!response.ok) {
            throw new Error(String(response.status));
          }
          return response.blob();
        })
        .then((blob) => {
          if (!signal.aborted) {
            setImageBlob(blob, item.name);
          }
        })
        .catch(() => {
          if (!signal.aborted) {
            setError(t('widgets.generate.promptTemplates.couldNotLoadImage'));
          }
        });
    },
    [abortGalleryFetch, setImageBlob, t]
  );

  const handleUploadFile = useCallback(
    (file: File) => {
      abortGalleryFetch();
      setImageBlob(file, file.name);
    },
    [abortGalleryFetch, setImageBlob]
  );

  const clearImage = useCallback(() => {
    abortGalleryFetch();
    takeObjectUrl(null);
    setDraft((current) => ({ ...current, image: { kind: 'remove' }, imageName: undefined, imagePreviewUrl: null }));
  }, [abortGalleryFetch, takeObjectUrl]);

  const handleSlotChange = useCallback(
    (item: GalleryItem | null) => {
      if (item === null) {
        clearImage();
      } else if (item.kind === 'image') {
        handleGalleryPick(item);
      }
    },
    [clearImage, handleGalleryPick]
  );
  const slotLabels = useMemo<Partial<GalleryMediaSlotLabels>>(
    () => ({
      choose: t('widgets.generate.promptTemplates.addImage'),
      remove: t('widgets.generate.promptTemplates.removeImage'),
      replace: t('widgets.generate.promptTemplates.replaceImage'),
    }),
    [t]
  );
  const slotThumbnail = useMemo(
    () => (
      <PromptTemplateImage
        alt=""
        boxSize="full"
        fallback={null}
        localPreviewUrl={draft.imagePreviewUrl}
        objectFit="contain"
        rounded="sm"
        template={template ?? NEW_TEMPLATE_IMAGE}
      />
    ),
    [draft.imagePreviewUrl, template]
  );
  const slotValue = useMemo<GalleryMediaSlotValue | null>(
    () =>
      hasImage
        ? { kind: 'image', name: draft.imageName ?? template?.name ?? t('widgets.generate.promptTemplates.image') }
        : null,
    [draft.imageName, hasImage, t, template?.name]
  );

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
        {/* Exclude template placeholders from dynamic highlighting because merging consumes them first. */}
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
        <GalleryMediaSlot
          accept={IMAGE_ONLY}
          dropId={IMAGE_DROP_ID}
          labels={slotLabels}
          thumbnail={slotThumbnail}
          value={slotValue}
          onChange={handleSlotChange}
          onUploadFile={handleUploadFile}
        />
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
