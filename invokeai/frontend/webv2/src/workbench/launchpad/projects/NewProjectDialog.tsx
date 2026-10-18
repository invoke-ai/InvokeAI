import type { GalleryBoard, GalleryImage, GalleryItem } from '@features/gallery';
/* eslint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { WorkflowLibraryListItem } from '@features/workflow/queries';
import type { LaunchpadIntentId } from '@workbench/launchpad/intents';
import type { LaunchpadStartEntry } from '@workbench/launchpad/startEntries';
import type { ProjectRecordDTO } from '@workbench/projects/api';
import type { FormEvent, KeyboardEvent, RefObject } from 'react';

import { Box, chakra, HStack, Icon, Input, Portal, Stack, Text } from '@chakra-ui/react';
import { GalleryMediaSlot } from '@features/gallery/mediaSlot';
import {
  galleryImageItemToGalleryImage,
  GalleryHostProvider,
  isGalleryImageItem,
  useGalleryUploadAction,
  type GalleryHost,
  type GalleryPickerAccept,
} from '@features/gallery/picker';
import {
  ensureArchitectureCapabilitiesLoaded,
  getArchitectureCapabilitiesSnapshot,
  subscribeArchitectureCapabilities,
  type ArchitectureCapabilitiesSnapshot,
} from '@features/generation/runtime';
import { isGenerateModelSelectable } from '@features/generation/settings';
import { useCapabilities } from '@features/identity';
import {
  ensureModelsLoaded,
  ModelsUiProvider,
  useModelsSelector,
  type ModelConfig,
  type ModelsUiAdapter,
  type ModelTaxonomyType,
} from '@features/models';
import { ModelSelect } from '@features/models/react';
import { WorkflowLibraryPicker } from '@features/workflow/launchpad';
import { readClipboardImage } from '@platform/browser/clipboardImage';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Button, CloseButton, IconButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { EmptyState } from '@platform/ui/EmptyState';
import { Field, FieldLabel } from '@platform/ui/Field';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { toaster } from '@platform/ui/toaster';
import { Tooltip } from '@platform/ui/Tooltip';
import { Link, useNavigate } from '@tanstack/react-router';
import { LAUNCHPAD_START_ENTRIES } from '@workbench/launchpad/startEntries';
import { requestNewProject } from '@workbench/projects/newProjectRequest';
import { useImportProjectFile } from '@workbench/projects/useProjectFileActions';
import {
  getWorkbenchPreferences,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
} from '@workbench/settings/store';
import { ArrowLeftRightIcon, BoxesIcon, ClipboardPasteIcon, FileUpIcon, SquareDashedIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  buildNewProjectRequest,
  commitDimension,
  createNewProjectDraft,
  getActiveAspectRatio,
  getStartImage,
  hasGenerationSettings,
  listSelectableGenerateModels,
  NEW_PROJECT_ASPECT_RATIOS,
  resolveNewProjectModel,
  resolveNewProjectSize,
  selectAspectRatio,
  selectModel,
  setCanvasStart,
  setStartImage,
  swapDimensions,
  type Dimensions,
  type NewProjectAspectRatioId,
  type NewProjectDraft,
  type NewProjectModel,
} from './newProjectDraft';

/**
 * One fixed frame for every start: the rail picks a start, the pane holds that start's choices, the footer names and
 * creates. Generate and Canvas starts choose a model and size here; Workflow picks a library workflow or a blank one;
 * the other starts explain what the editor asks for.
 */

/** Called as the dialog opens: both catalogs are shared, so a second visit finds them loaded. */
export const prepareNewProjectDialog = (): void => {
  void ensureModelsLoaded();
  void ensureArchitectureCapabilitiesLoaded();
};

const DIALOG_WIDTH = 'min(880px, calc(100vw - 2rem))';
const DIALOG_HEIGHT = '560px';
const DIALOG_MAX_HEIGHT = 'calc(100dvh - 4rem)';
const RAIL_WIDTH = '220px';
const PREVIEW_STAGE_PX = 96;
const PREVIEW_PAD_PX = 8;
const MODELS_ROUTE_SEARCH = {} as const;
const GENERATE_MODEL_TYPES: readonly ModelTaxonomyType[] = ['main', 'external_image_generator'];
const IMAGE_ONLY: GalleryPickerAccept = ['image'];
/** The Launchpad has no project board to seed the picker from and no Gallery widget to reveal. */
const NO_GALLERY_VALUES: Record<string, unknown> = {};
const NO_BOARDS: GalleryBoard[] = [];
/** Pasted images land unfiled, where the slot's own uploads go without a project board. */
const UPLOAD_BOARD_ID = 'none';
const GALLERY_HOST: GalleryHost = {
  galleryValues: NO_GALLERY_VALUES,
  notifications: {
    add: ({ kind, message, title }) => toaster.create({ description: message, title, type: kind }),
    reportError: ({ message }) => toaster.create({ title: message, type: 'error' }),
  },
  projectName: '',
};

const selectCapabilitiesView = (snapshot: ArchitectureCapabilitiesSnapshot) => ({
  error: snapshot.error,
  status: snapshot.status,
});

/** The pasted file keeps the clipboard's encoding; the gallery names it by the extension. */
const getClipboardFileName = (type: string): string =>
  type === 'image/webp' ? 'clipboard.webp' : type === 'image/jpeg' ? 'clipboard.jpg' : 'clipboard.png';

const START_IDS = LAUNCHPAD_START_ENTRIES.map((entry) => entry.id);

const getAdjacentStart = (current: LaunchpadIntentId, key: string): LaunchpadIntentId | null => {
  if (key === 'Home') {
    return START_IDS[0] ?? null;
  }

  if (key === 'End') {
    return START_IDS.at(-1) ?? null;
  }

  const direction = key === 'ArrowDown' ? 1 : key === 'ArrowUp' ? -1 : 0;

  if (direction === 0) {
    return null;
  }

  const index = START_IDS.indexOf(current);

  return START_IDS[(Math.max(index, 0) + direction + START_IDS.length) % START_IDS.length] ?? null;
};

const SectionHeading = ({ children, id }: { children: string; id?: string }) => (
  <Text color="fg.muted" fontSize="xs" fontWeight="700" id={id} textTransform="uppercase">
    {children}
  </Text>
);

const StartOption = ({
  entry,
  isSelected,
  ref,
  onSelect,
}: {
  entry: LaunchpadStartEntry;
  isSelected: boolean;
  ref: RefObject<HTMLButtonElement | null> | undefined;
  onSelect: (intent: LaunchpadIntentId) => void;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => onSelect(entry.id), [entry.id, onSelect]);

  return (
    <chakra.button
      ref={ref}
      alignItems="center"
      aria-selected={isSelected}
      borderRadius="md"
      color={isSelected ? 'fg' : 'fg.muted'}
      data-intent={entry.id}
      display="flex"
      fontWeight={isSelected ? '600' : '500'}
      gap="2.5"
      h="8"
      px="2.5"
      role="option"
      tabIndex={isSelected ? 0 : -1}
      textAlign="start"
      transitionDuration="var(--wb-motion-duration-fast)"
      transitionProperty="background, color"
      type="button"
      w="full"
      _focusVisible={{ outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '-2px' }}
      _hover={{ bg: 'bg.muted', color: 'fg' }}
      _selected={{ bg: 'bg.emphasized' }}
      onClick={handleClick}
    >
      <Icon as={entry.icon} boxSize="4" flexShrink="0" />
      <Text as="span" truncate>
        {t(entry.labelKey)}
      </Text>
    </chakra.button>
  );
};

/** Typed values commit on blur or Enter; Enter stays here so a snapped value is seen before the project is created. */
const DimensionInput = ({
  disabled,
  value,
  onCommit,
}: {
  disabled: boolean;
  value: number;
  onCommit: (value: number) => void;
}) => {
  const [text, setText] = useState<string | null>(null);
  const commit = (raw: string) => {
    setText(null);
    onCommit(Number.parseInt(raw, 10));
  };

  return (
    <Input
      disabled={disabled}
      inputMode="numeric"
      size="sm"
      textAlign="end"
      value={text ?? String(value)}
      // Only typed text commits: tabbing through must not turn a chip or default size into a custom one.
      onBlur={(event) => {
        if (text !== null) {
          commit(event.currentTarget.value);
        }
      }}
      onChange={(event) => setText(event.currentTarget.value)}
      onKeyDown={(event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          commit(event.currentTarget.value);
        }
      }}
    />
  );
};

const SizePreview = ({ size }: { size: Dimensions }) => {
  const { t } = useTranslation();
  const scale = (PREVIEW_STAGE_PX - PREVIEW_PAD_PX * 2) / Math.max(size.width, size.height, 1);

  return (
    <Stack align="center" flexShrink="0" gap="1.5">
      <Box
        aria-hidden="true"
        bg="bg.muted"
        borderColor="border.subtle"
        borderWidth="1px"
        boxSize={`${PREVIEW_STAGE_PX}px`}
        position="relative"
        rounded="sm"
      >
        <Box
          bg="bg.emphasized"
          borderColor="border.emphasized"
          borderWidth="1.5px"
          h={`${size.height * scale}px`}
          left="50%"
          position="absolute"
          rounded="2px"
          top="50%"
          transform="translate(-50%, -50%)"
          transitionDuration="var(--wb-motion-duration-fast)"
          transitionProperty="width, height"
          w={`${size.width * scale}px`}
        />
      </Box>
      <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums">
        {t('projects.newProjectDialog.sizeValue', { height: size.height, width: size.width })}
      </Text>
    </Stack>
  );
};

const GenerationFields = ({
  canvasStart,
  model,
  size,
  activeRatio,
  startImage,
  onBusyChange,
  onCanvasStartChange,
  onModelChange,
  onRatioSelect,
  onDimensionCommit,
  onStartImageChange,
  onSwap,
}: {
  /** Null for a Generate start, which has no Start-from control. */
  canvasStart: 'blank' | 'image' | null;
  model: NewProjectModel | null;
  size: Dimensions;
  activeRatio: NewProjectAspectRatioId | null;
  startImage: GalleryImage | null;
  onBusyChange: (isBusy: boolean) => void;
  onCanvasStartChange: (start: 'blank' | 'image') => void;
  onModelChange: (model: NewProjectModel) => void;
  onRatioSelect: (ratioId: NewProjectAspectRatioId) => void;
  onDimensionCommit: (key: keyof Dimensions, value: number) => void;
  onStartImageChange: (image: GalleryImage | null) => void;
  onSwap: () => void;
}) => {
  const { t } = useTranslation();
  const handleModelChange = useCallback(
    (next: ModelConfig | null) => {
      if (next && isGenerateModelSelectable(next)) {
        onModelChange(next);
      }
    },
    [onModelChange]
  );
  const swapLabel = t('projects.newProjectDialog.swapDimensions');
  // A chosen image sizes the canvas, so the size controls show its dimensions and rest.
  const isSizeLocked = startImage !== null;

  return (
    <>
      <Field flex="0 0 auto" label={t('projects.newProjectDialog.model')} maxW="28rem">
        <ModelSelect
          filter={isGenerateModelSelectable}
          modelTypes={GENERATE_MODEL_TYPES}
          value={model?.key ?? null}
          onChange={handleModelChange}
        />
      </Field>

      {/* The image choice drives the size, so it comes first on the Canvas start. */}
      {canvasStart !== null ? (
        <StartFromSection
          canvasStart={canvasStart}
          startImage={startImage}
          onBusyChange={onBusyChange}
          onCanvasStartChange={onCanvasStartChange}
          onStartImageChange={onStartImageChange}
        />
      ) : null}

      <Stack flexShrink="0" gap="1.5">
        <FieldLabel>{t('projects.newProjectDialog.size')}</FieldLabel>
        <HStack align="start" gap="6">
          <Stack gap="2" minW="0">
            <HStack aria-label={t('projects.newProjectDialog.aspectRatio')} flexWrap="wrap" gap="1.5" role="group">
              {NEW_PROJECT_ASPECT_RATIOS.map((ratioId) => {
                const isActive = ratioId === activeRatio;

                return (
                  <Button
                    key={ratioId}
                    aria-pressed={isActive}
                    colorPalette={isActive ? 'accent' : 'gray'}
                    disabled={isSizeLocked}
                    fontVariantNumeric="tabular-nums"
                    minW="11"
                    size="xs"
                    type="button"
                    variant={isActive ? 'solid' : 'outline'}
                    onClick={() => onRatioSelect(ratioId)}
                  >
                    {ratioId}
                  </Button>
                );
              })}
            </HStack>
            <HStack align="end" gap="2">
              <Field disabled={isSizeLocked} flex="0 0 auto" label={t('projects.newProjectDialog.width')} w="6rem">
                <DimensionInput
                  disabled={isSizeLocked}
                  value={size.width}
                  onCommit={(value) => onDimensionCommit('width', value)}
                />
              </Field>
              <Tooltip content={swapLabel}>
                <IconButton
                  aria-label={swapLabel}
                  disabled={isSizeLocked}
                  mb="0.5"
                  size="sm"
                  type="button"
                  variant="ghost"
                  onClick={onSwap}
                >
                  <ArrowLeftRightIcon />
                </IconButton>
              </Tooltip>
              <Field disabled={isSizeLocked} flex="0 0 auto" label={t('projects.newProjectDialog.height')} w="6rem">
                <DimensionInput
                  disabled={isSizeLocked}
                  value={size.height}
                  onCommit={(value) => onDimensionCommit('height', value)}
                />
              </Field>
            </HStack>
          </Stack>
          <SizePreview size={size} />
        </HStack>
      </Stack>
    </>
  );
};

/**
 * Blank, or an image the canvas opens on. The slot uploads files and picks from the gallery itself; the clipboard
 * button pushes a pasted image through the picker's own upload action, so Create always holds a server image.
 */
const StartFromSection = ({
  canvasStart,
  startImage,
  onBusyChange,
  onCanvasStartChange,
  onStartImageChange,
}: {
  canvasStart: 'blank' | 'image';
  startImage: GalleryImage | null;
  /** The clipboard upload is in flight; Create waits for it. */
  onBusyChange: (isBusy: boolean) => void;
  onCanvasStartChange: (start: 'blank' | 'image') => void;
  onStartImageChange: (image: GalleryImage | null) => void;
}) => {
  const { t } = useTranslation();
  const startOptions = useMemo(
    () => [
      { label: t('projects.newProjectDialog.startBlank'), value: 'blank' },
      { label: t('projects.newProjectDialog.startImage'), value: 'image' },
    ],
    [t]
  );
  const handleStartChange = useCallback(
    (value: string) => onCanvasStartChange(value === 'image' ? 'image' : 'blank'),
    [onCanvasStartChange]
  );

  return (
    <Stack flexShrink="0" gap="1.5">
      <FieldLabel>{t('projects.newProjectDialog.startFrom')}</FieldLabel>
      <Box maxW="16rem">
        <SegmentedControl
          ariaLabel={t('projects.newProjectDialog.startFrom')}
          options={startOptions}
          value={canvasStart}
          onChange={handleStartChange}
        />
      </Box>
      {canvasStart === 'image' ? (
        <GalleryHostProvider host={GALLERY_HOST}>
          <StartImagePicker
            startImage={startImage}
            onBusyChange={onBusyChange}
            onStartImageChange={onStartImageChange}
          />
        </GalleryHostProvider>
      ) : null}
    </Stack>
  );
};

/** Inside the gallery host: the upload action reports through it, as the slot's picker does. */
const StartImagePicker = ({
  startImage,
  onBusyChange,
  onStartImageChange,
}: {
  startImage: GalleryImage | null;
  onBusyChange: (isBusy: boolean) => void;
  onStartImageChange: (image: GalleryImage | null) => void;
}) => {
  const { t } = useTranslation();
  const uploadFiles = useGalleryUploadAction({ boards: NO_BOARDS, selectedBoardId: UPLOAD_BOARD_ID });
  const { isBusy, run } = useScopedAction();
  // The last clipboard attempt found nothing; cleared by the next attempt or a chosen image.
  const [isClipboardEmpty, setIsClipboardEmpty] = useState(false);
  const slotValue = useMemo(
    () =>
      startImage
        ? { height: startImage.height, kind: 'image' as const, name: startImage.imageName, width: startImage.width }
        : null,
    [startImage]
  );
  const handleSlotChange = useCallback(
    (item: GalleryItem | null) => {
      setIsClipboardEmpty(false);
      onStartImageChange(item && isGalleryImageItem(item) ? galleryImageItemToGalleryImage(item) : null);
    },
    [onStartImageChange]
  );
  const handleClipboard = () => {
    setIsClipboardEmpty(false);
    onBusyChange(true);
    void run(async (owner) => {
      const blob = await readClipboardImage();

      assertAccountScopeCurrent(owner);

      if (!blob) {
        setIsClipboardEmpty(true);
        return;
      }

      // Failures are reported by the action itself; an empty result is one it has already explained.
      const [item] = await uploadFiles([new File([blob], getClipboardFileName(blob.type), { type: blob.type })]);

      assertAccountScopeCurrent(owner);

      if (item && isGalleryImageItem(item)) {
        onStartImageChange(galleryImageItemToGalleryImage(item));
      }
    }).finally(() => onBusyChange(false));
  };

  return (
    <HStack align="start" gap="4">
      <Box flexShrink={0} w="24">
        <GalleryMediaSlot
          accept={IMAGE_ONLY}
          busy={isBusy}
          dropId="new-project-start-image"
          layout="tile"
          value={slotValue}
          onChange={handleSlotChange}
        />
      </Box>
      <Stack align="start" gap="2" pt="1">
        <Button disabled={isBusy} size="sm" type="button" variant="outline" onClick={handleClipboard}>
          <Icon as={ClipboardPasteIcon} boxSize="3.5" />
          {t('projects.newProjectDialog.fromClipboard')}
        </Button>
        {isClipboardEmpty ? (
          <Text color="fg.muted" fontSize="xs" role="status">
            {t('projects.newProjectDialog.clipboardEmpty')}
          </Text>
        ) : startImage ? (
          <Text color="fg.muted" fontSize="xs">
            {t('projects.newProjectDialog.imageSizeHint')}
          </Text>
        ) : null}
      </Stack>
    </HStack>
  );
};

/** The first row of the Workflow pane: a Create with nothing picked still opens one blank workflow. */
const BlankWorkflowOption = ({ isSelected, onSelect }: { isSelected: boolean; onSelect: () => void }) => {
  const { t } = useTranslation();

  return (
    <chakra.button
      alignItems="center"
      aria-pressed={isSelected}
      bg={isSelected ? 'bg.emphasized' : 'bg.subtle'}
      borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
      borderRadius="lg"
      borderWidth="1px"
      display="flex"
      flexShrink="0"
      gap="3"
      px="3"
      py="2"
      textAlign="start"
      transitionDuration="var(--wb-motion-duration-fast)"
      transitionProperty="background, border-color"
      type="button"
      w="full"
      _focusVisible={{ outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '2px' }}
      _hover={{ bg: 'bg.muted' }}
      onClick={onSelect}
    >
      <Icon as={SquareDashedIcon} boxSize="4" color="fg.muted" flexShrink="0" />
      <Stack gap="0" minW="0">
        <Text fontWeight="600">{t('projects.newProjectDialog.blankWorkflow')}</Text>
        <Text color="fg.muted" fontSize="xs">
          {t('projects.newProjectDialog.blankWorkflowDescription')}
        </Text>
      </Stack>
    </chakra.button>
  );
};

const EditorHint = ({ intent }: { intent: LaunchpadIntentId }) => {
  const { t } = useTranslation();

  return (
    <Text color="fg.muted">
      {intent === 'upscale' ? t('projects.newProjectDialog.upscaleHint') : t('projects.newProjectDialog.videoHint')}
    </Text>
  );
};

const NewProjectContent = ({
  selectedStartRef,
  onClose,
}: {
  selectedStartRef: RefObject<HTMLButtonElement | null>;
  onClose: () => void;
}) => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { canManageModels } = useCapabilities();
  // Read once per open: the dialog remembers the last start and model, then follows the user's picks.
  const [rememberedModelKey] = useState(() => getWorkbenchPreferences().launchpadNewProjectModelKey);
  const [draft, setDraft] = useState(() => createNewProjectDraft(getWorkbenchPreferences().launchpadNewProjectIntent));
  const [isStartImageBusy, setIsStartImageBusy] = useState(false);
  const startHeadingId = useId();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const catalogStatus = useModelsSelector((snapshot) => snapshot.status);
  // Grids, optimal sizes and the model defaults come from the served capability table; the size re-resolves when
  // it lands, and Create waits for it so fallback defaults never persist as a system patch.
  const capabilities = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    selectCapabilitiesView
  );

  const selectableModels = useMemo(() => listSelectableGenerateModels(models), [models]);
  const model = resolveNewProjectModel(draft, selectableModels, rememberedModelKey);
  const size = resolveNewProjectSize(draft, model);
  const entry =
    LAUNCHPAD_START_ENTRIES.find((candidate) => candidate.id === draft.intent) ?? LAUNCHPAD_START_ENTRIES[0]!;
  const needsModel = hasGenerationSettings(draft.intent);
  const isCatalogPending = catalogStatus === 'idle' || catalogStatus === 'loading';
  const hasNoModels = catalogStatus === 'loaded' && selectableModels.length === 0;
  // ModelSelect reports loading and load errors in its own list; the footer says why Create waits.
  const isCapabilitiesPending = capabilities.status === 'idle' || capabilities.status === 'loading';
  const isCanvasImageStart = draft.intent === 'canvas' && draft.canvasStart === 'image';
  const createBlockedReason = !needsModel
    ? null
    : model === null
      ? isCatalogPending
        ? t('projects.newProjectDialog.modelLoading')
        : catalogStatus === 'error'
          ? t('projects.newProjectDialog.modelLoadFailed')
          : t('projects.newProjectDialog.noModelsTitle')
      : isCapabilitiesPending
        ? t('common.loading')
        : capabilities.status === 'error'
          ? t('projects.newProjectDialog.capabilitiesLoadFailed')
          : isStartImageBusy
            ? t('common.loading')
            : isCanvasImageStart && draft.startImage === null
              ? t('projects.newProjectDialog.chooseImageOrBlank')
              : null;
  const canRetryCapabilities = needsModel && model !== null && capabilities.status === 'error';
  const handleRetryCapabilities = useCallback(() => void ensureArchitectureCapabilitiesLoaded(), []);

  const openImportedProject = useCallback(
    async (record: ProjectRecordDTO) => {
      await navigate({ search: { project: record.project_id }, to: '/app' });
    },
    [navigate]
  );
  const importFile = useImportProjectFile(openImportedProject);
  const handleImport = useCallback(() => {
    onClose();
    importFile();
  }, [importFile, onClose]);

  // Creation reads the draft it is given, so a pick that creates in the same gesture is never a render behind.
  const create = (candidate: NewProjectDraft) => {
    if (createBlockedReason !== null) {
      return;
    }

    const request = buildNewProjectRequest({ draft: candidate, model, models });

    // The model preference belongs to the Generate and Canvas starts; other starts leave it as it was.
    void patchWorkbenchPreferences({
      launchpadNewProjectIntent: candidate.intent,
      ...(needsModel ? { launchpadNewProjectModelKey: model?.key ?? null } : {}),
    });
    requestNewProject(request);
    void navigate({ search: { intent: candidate.intent, new: true }, to: '/app' });
    onClose();
  };
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    create(draft);
  };

  const handleStartSelect = useCallback(
    (intent: LaunchpadIntentId) => setDraft((current) => ({ ...current, intent })),
    []
  );
  const handleRailKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      create(draft);

      return;
    }

    const next = getAdjacentStart(draft.intent, event.key);

    if (!next) {
      return;
    }

    event.preventDefault();
    setDraft((current) => ({ ...current, intent: next }));
    event.currentTarget.querySelector<HTMLButtonElement>(`[data-intent="${next}"]`)?.focus();
  };
  const handleModelChange = useCallback(
    (next: NewProjectModel) => setDraft((current) => selectModel(current, next)),
    []
  );
  // The size handlers close over the resolved model; the React Compiler memoizes them, manual deps cannot.
  const handleRatioSelect = (ratioId: NewProjectAspectRatioId) =>
    setDraft((current) => selectAspectRatio(current, ratioId, model));
  const handleDimensionCommit = (key: keyof Dimensions, value: number) =>
    setDraft((current) => commitDimension(current, model, key, value));
  const handleSwap = () => setDraft((current) => swapDimensions(current, model));
  const handleCanvasStartChange = useCallback(
    (start: 'blank' | 'image') => setDraft((current) => setCanvasStart(current, start)),
    []
  );
  const handleStartImageChange = useCallback(
    (image: GalleryImage | null) => setDraft((current) => setStartImage(current, image)),
    []
  );
  const handleBlankWorkflowSelect = useCallback(
    () => setDraft((current) => ({ ...current, libraryWorkflow: null })),
    []
  );
  const handleWorkflowSelect = useCallback(
    (item: WorkflowLibraryListItem) =>
      setDraft((current) => ({ ...current, libraryWorkflow: { name: item.name, workflowId: item.workflow_id } })),
    []
  );
  const handleWorkflowOpen = (item: WorkflowLibraryListItem) =>
    create({ ...draft, libraryWorkflow: { name: item.name, workflowId: item.workflow_id } });

  return (
    <Dialog.Content h={DIALOG_HEIGHT} maxH={DIALOG_MAX_HEIGHT} maxW={DIALOG_WIDTH} overflow="hidden" w="full">
      <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle" pb="3">
        <Dialog.Title>{t('projects.newProjectDialog.title')}</Dialog.Title>
        <Dialog.CloseTrigger asChild>
          <CloseButton type="button" />
        </Dialog.CloseTrigger>
      </Dialog.Header>
      <Dialog.Body display="flex" flex="1" minH="0" p="0">
        <Stack
          borderColor="border.subtle"
          borderRightWidth="1px"
          flexShrink="0"
          gap="4"
          overflowY="auto"
          p="3"
          w={RAIL_WIDTH}
        >
          <Stack gap="1.5">
            <SectionHeading id={startHeadingId}>{t('projects.newProjectDialog.startWith')}</SectionHeading>
            <Stack
              aria-labelledby={startHeadingId}
              gap="0.5"
              role="listbox"
              tabIndex={-1}
              onKeyDown={handleRailKeyDown}
            >
              {LAUNCHPAD_START_ENTRIES.map((start) => (
                <StartOption
                  key={start.id}
                  entry={start}
                  isSelected={start.id === draft.intent}
                  ref={start.id === draft.intent ? selectedStartRef : undefined}
                  onSelect={handleStartSelect}
                />
              ))}
            </Stack>
          </Stack>
          <Stack borderColor="border.subtle" borderTopWidth="1px" gap="1.5" pt="3">
            <SectionHeading>{t('projects.newProjectDialog.from')}</SectionHeading>
            <Button
              color="fg.muted"
              fontWeight="500"
              justifyContent="flex-start"
              px="2.5"
              type="button"
              variant="ghost"
              onClick={handleImport}
            >
              <Icon as={FileUpIcon} boxSize="4" />
              {t('projects.newProjectDialog.importFile')}
            </Button>
          </Stack>
        </Stack>

        <Stack flex="1" gap="4" minH="0" minW="0" overflow="hidden" p="5">
          <Stack flexShrink="0" gap="0.5">
            <Text as="h3" fontWeight="600" textStyle="lg">
              {t(entry.labelKey)}
            </Text>
            <Text color="fg.muted">{t(entry.descriptionKey)}</Text>
          </Stack>
          {needsModel ? (
            hasNoModels ? (
              <EmptyState
                align="start"
                description={t('projects.newProjectDialog.noModelsDescription')}
                icon={<BoxesIcon />}
                title={t('projects.newProjectDialog.noModelsTitle')}
              >
                {canManageModels ? (
                  <Button asChild size="sm" variant="outline">
                    <Link search={MODELS_ROUTE_SEARCH} to="/models" onClick={onClose}>
                      {t('projects.newProjectDialog.openModelManager')}
                    </Link>
                  </Button>
                ) : null}
              </EmptyState>
            ) : (
              <Stack flex="1" gap="4" minH="0" overflowY="auto">
                <GenerationFields
                  activeRatio={getActiveAspectRatio(draft, size)}
                  canvasStart={draft.intent === 'canvas' ? draft.canvasStart : null}
                  model={model}
                  size={size}
                  startImage={getStartImage(draft)}
                  onBusyChange={setIsStartImageBusy}
                  onCanvasStartChange={handleCanvasStartChange}
                  onDimensionCommit={handleDimensionCommit}
                  onModelChange={handleModelChange}
                  onRatioSelect={handleRatioSelect}
                  onStartImageChange={handleStartImageChange}
                  onSwap={handleSwap}
                />
              </Stack>
            )
          ) : draft.intent === 'workflow' ? (
            <>
              <BlankWorkflowOption isSelected={draft.libraryWorkflow === null} onSelect={handleBlankWorkflowSelect} />
              <WorkflowLibraryPicker
                selectedWorkflowId={draft.libraryWorkflow?.workflowId ?? null}
                onOpen={handleWorkflowOpen}
                onSelect={handleWorkflowSelect}
              />
            </>
          ) : (
            <EditorHint intent={draft.intent} />
          )}
        </Stack>
      </Dialog.Body>
      {/* Only the footer is a form: the panes hold buttons of their own that must never submit it. */}
      <Dialog.Footer
        asChild
        borderColor="border.subtle"
        borderTopWidth="1px"
        flexWrap="wrap"
        justifyContent="space-between"
        pt="3"
      >
        <chakra.form onSubmit={handleSubmit}>
          <Field flex="1" label={t('common.name')} maxW="22rem" minW="14rem" orientation="horizontal">
            <Input
              autoComplete="off"
              name="projectName"
              placeholder={t('projects.newProjectDialog.namePlaceholder')}
              size="md"
              value={draft.name}
              onChange={(event) => {
                const name = event.currentTarget.value;

                setDraft((current) => ({ ...current, name }));
              }}
            />
          </Field>
          <HStack gap="3">
            {createBlockedReason ? (
              <Text color="fg.muted" fontSize="xs">
                {createBlockedReason}
              </Text>
            ) : null}
            {canRetryCapabilities ? (
              <Button size="xs" type="button" variant="ghost" onClick={handleRetryCapabilities}>
                {t('common.retry')}
              </Button>
            ) : null}
            <HStack gap="2">
              <Button type="button" variant="ghost" onClick={onClose}>
                {t('common.cancel')}
              </Button>
              <Button disabled={createBlockedReason !== null} type="submit" variant="solid">
                {t('projects.newProjectDialog.create')}
              </Button>
            </HStack>
          </HStack>
        </chakra.form>
      </Dialog.Footer>
    </Dialog.Content>
  );
};

export const NewProjectDialog = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const { canManageModels } = useCapabilities();
  const enableModelDescriptions = useWorkbenchPreferenceSelector((preferences) => preferences.enableModelDescriptions);
  const selectedStartRef = useRef<HTMLButtonElement | null>(null);
  // Land on the selected start, so arrow keys and Enter work before anything else is touched.
  const getInitialFocusEl = useCallback(() => selectedStartRef.current, []);
  // No project is open in the Launchpad: the model picker's manager link needs no project to return to.
  const modelsUiAdapter = useMemo<ModelsUiAdapter>(
    () => ({ canManageModels, enableModelDescriptions, isProjectActive: () => true, managerProjectId: null }),
    [canManageModels, enableModelDescriptions]
  );
  const handleOpenChange = useCallback(
    (event: Dialog.OpenChangeDetails) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Dialog.Root
      initialFocusEl={getInitialFocusEl}
      lazyMount
      open={isOpen}
      placement="center"
      size="xl"
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <ModelsUiProvider adapter={modelsUiAdapter}>
            <NewProjectContent selectedStartRef={selectedStartRef} onClose={onClose} />
          </ModelsUiProvider>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
