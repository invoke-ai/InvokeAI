import type { ModelConfig, ModelTaxonomyType } from '@features/models/react';
import type { FieldInputTemplate } from '@features/workflow/contracts';
import type { LoraFieldCollectionEntry } from '@features/workflow/utility';
import type { SeedInputPatch } from '@platform/ui/SeedInput';

import {
  Badge,
  Box,
  createListCollection,
  Field,
  Flex,
  HStack,
  Icon,
  Image,
  Input,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  chakra,
} from '@chakra-ui/react';
import { useDndContext, useDndMonitor, useDroppable, type DragEndEvent } from '@dnd-kit/core';
import {
  formatGalleryVideoDuration,
  galleryDestinations,
  galleryItems,
  galleryTransfers,
  type GalleryBoard,
  type GalleryItem,
} from '@features/gallery';
import { getSelectedGalleryImageFromValues, toGalleryItemKey } from '@features/gallery/contracts';
import { GalleryPickerPopover, type GalleryPickerSelection } from '@features/gallery/picker';
import { invalidateGallery } from '@features/gallery/queries';
import { galleryImageUrls, galleryVideoUrls } from '@features/gallery/utility';
import { DEFAULT_LORA_WEIGHT_CONFIG, sanitizeBatchCount, SCHEDULER_OPTIONS } from '@features/generation/settings';
import { isInvocationNode } from '@features/workflow/contracts';
import {
  buildSavedWorkflowOptions,
  getSavedWorkflowDisplayState,
  getSavedWorkflowListItemFromRecord,
  getSavedWorkflowPickerOwnedQuery,
  getSavedWorkflowPickerSharedQuery,
  getSavedWorkflowSelectionOption,
  getSavedWorkflowSelectionState,
  mergeSavedWorkflowPickerItems,
  MISSING_WORKFLOW_OPTION_VALUE,
  shouldFetchNextSavedWorkflowPickerPage,
} from '@features/workflow/data/savedWorkflowFieldUtils';
import {
  getWorkflowPagesItems,
  savedWorkflowDetailQueryOptions,
  savedWorkflowPickerQueryOptions,
} from '@features/workflow/data/savedWorkflowQueries';
import { isSeedInputField } from '@features/workflow/graph';
import {
  getWorkflowMediaFieldDropId,
  getWorkflowMediaFieldDropItem,
  getWorkflowMediaFieldDropItems,
  type WorkflowMediaKind,
} from '@features/workflow/ui/fields/mediaFieldDnd';
import {
  finiteNumberOrUndefined,
  invalidProps,
  NumericInput,
  useFocusedDraft,
} from '@features/workflow/ui/fields/NumericInput';
import { useWorkflowProjectSelector, useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import {
  getResolvedWorkflowEdges,
  isLoraFieldCollectionEntry,
  isLoraFieldWeightValid,
  isWorkflowCollectionItemValid,
  isWorkflowGeneratorFieldTypeName,
  LORA_FIELD_WEIGHT_RANGE,
  toLoraFieldCollectionList,
} from '@features/workflow/utility';
import { planSeedSubmission, type SeedMode, wrapSeed } from '@platform/core/seed';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import {
  Button,
  ColorPicker,
  Combobox,
  DropTargetOverlay,
  formatHexColor,
  IconButton,
  parseHexColor,
  ResizableTextarea,
  Select,
  Slider,
  toaster,
  Tooltip,
} from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { SeedInput } from '@platform/ui/SeedInput';
import { useInfiniteQuery, useQuery, useQueryClient } from '@tanstack/react-query';
import { FilmIcon, ImageIcon, ImagePlusIcon, PlusIcon, RotateCcwIcon, Trash2Icon, XIcon } from 'lucide-react';
import {
  lazy,
  Suspense,
  useCallback,
  useDeferredValue,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from 'react';
import { useTranslation } from 'react-i18next';

const ModelSelect = lazy(() => import('@features/models/react').then((module) => ({ default: module.ModelSelect })));
const MODEL_SELECT_FALLBACK = (
  <Button disabled size="xs" w="full">
    Loading models…
  </Button>
);
const RECORD_PICKER_FALLBACK = (
  <Button disabled size="xs" w="full">
    Loading…
  </Button>
);
// Generator settings load with their node; a plain workflow never pays for them.
const GeneratorFieldInput = lazy(() =>
  import('./GeneratorFieldInput').then((module) => ({ default: module.GeneratorFieldInput }))
);

// Record pickers load with their node so the system-prompt query stays out of the editor's boot graph.
const StylePresetInput = lazy(() =>
  import('./RecordPickerInput').then((module) => ({ default: module.StylePresetInput }))
);
const SystemPromptInput = lazy(() =>
  import('./RecordPickerInput').then((module) => ({ default: module.SystemPromptInput }))
);

export const getWorkflowSelectedGalleryImage = getSelectedGalleryImageFromValues;

export interface WorkflowFieldInputProps {
  id?: string;
  invalid?: boolean;
  /** Owning invocation node, when known — lets widgets read sibling fields (e.g. the frame scrubber's companion video). */
  nodeId?: string;
  template: FieldInputTemplate;
  value: unknown;
  onChange: (value: unknown) => void;
  /** The instance's seed mode; read only for seed inputs (`isSeedInputField`), which render the mode menu. */
  seedMode?: SeedMode;
  onSeedModeChange?: (seedMode: SeedMode) => void;
}

// The media well's hover, matching DropZone's pointer-hover accent preview.
const MEDIA_INPUT_HOVER_PROPS = { borderColor: 'accent.solid' };

/** A row of a list names itself by position; a scalar field is named by its title. */
type ScalarInputProps = WorkflowFieldInputProps & { ariaLabel?: string };

const StringInput = ({ ariaLabel, id, invalid, onChange, template, value }: ScalarInputProps) => {
  const [draft, setDraft, clearDraft] = useFocusedDraft();
  const text = draft ?? (typeof value === 'string' ? value : '');
  const onTextChange = useCallback(
    (event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setDraft(event.currentTarget.value);
      onChange(event.currentTarget.value);
    },
    [onChange, setDraft]
  );

  if (template.uiComponent === 'textarea') {
    return (
      <ResizableTextarea
        aria-label={ariaLabel ?? template.title}
        className="nodrag nowheel"
        defaultHeightPx={96}
        fontFamily="mono"
        id={id ? `${id}-textarea` : undefined}
        minHeightPx={56}
        resizeHandleAriaLabel={`Resize ${template.title}`}
        size="xs"
        value={text}
        w="full"
        {...invalidProps(invalid)}
        onBlur={clearDraft}
        onChange={onTextChange}
      />
    );
  }

  return (
    <Input
      aria-label={ariaLabel ?? template.title}
      className="nodrag"
      id={id ? `${id}-input` : undefined}
      size="xs"
      value={text}
      w="full"
      {...invalidProps(invalid)}
      onBlur={clearDraft}
      onChange={onTextChange}
    />
  );
};

/**
 * Plan stepping previews from workflow iterations and template defaults; nokey prevents node shortcuts inside the
 * row and portalled menu.
 */
const WorkflowSeedInput = ({
  id,
  invalid,
  onChange,
  onSeedModeChange,
  seedMode,
  template,
  value,
}: WorkflowFieldInputProps & { onSeedModeChange: (seedMode: SeedMode) => void; seedMode: SeedMode }) => {
  const { t } = useTranslation();
  const batchCount = useWorkflowProjectSelector((project) => sanitizeBatchCount(project.workflowValues.batchCount));
  const seed = typeof value === 'number' ? value : undefined;
  const authoredSeed = seed ?? (typeof template.default === 'number' ? template.default : 0);
  const plan =
    seedMode === 'increment' || seedMode === 'decrement'
      ? planSeedSubmission({
          batchCount,
          promptCount: 1,
          seedBehaviour: 'per-iteration',
          seedMode,
          startSeed: wrapSeed(authoredSeed),
        })
      : null;
  const onCommit = useCallback(
    (patch: SeedInputPatch) => {
      if (patch.seed !== undefined) {
        onChange(patch.seed);
      }

      if (patch.seedMode !== undefined) {
        onSeedModeChange(patch.seedMode);
      }
    },
    [onChange, onSeedModeChange]
  );

  return (
    <SeedInput
      ariaLabel={template.title}
      className="nodrag nokey"
      contentClassName="nokey"
      description={t('nodes.seedModeTooltip')}
      id={id ? `${id}-number-input` : undefined}
      invalid={invalid}
      plan={plan}
      seed={seed}
      seedMode={seedMode}
      onCommit={onCommit}
    />
  );
};

const SWITCH_CHECKED_PROPS = { bg: 'accent.solid' };

const BooleanInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const onCheckedChange = useCallback((event: { checked: boolean }) => onChange(event.checked), [onChange]);
  // The host id goes through zag's id map: the root is a `<label for>` pointing at zag's hidden-input
  // id, so an `id` set on the element itself would leave the label pointing at nothing (an inert switch).
  const switchIds = useMemo(() => (id ? { hiddenInput: `${id}-switch-input` } : undefined), [id]);

  return (
    <Switch.Root
      checked={value === true}
      className="nodrag"
      ids={switchIds}
      invalid={invalid}
      size="sm"
      onCheckedChange={onCheckedChange}
    >
      <Switch.HiddenInput aria-label={template.title} {...invalidProps(invalid)} />
      <Switch.Control _checked={SWITCH_CHECKED_PROPS}>
        <Switch.Thumb />
      </Switch.Control>
    </Switch.Root>
  );
};

const SELECT_VALUE_TEXT_PROPS = { placeholder: 'Select…' };

const SelectInput = ({
  id,
  invalid,
  onChange,
  options,
  title,
  value,
}: {
  id?: string;
  onChange: (value: string) => void;
  invalid?: boolean;
  options: { label: string; value: string }[];
  title: string;
  value: unknown;
}) => {
  const collection = useMemo(() => createListCollection({ items: options }), [options]);
  const selectedValue = useMemo(() => {
    const key =
      typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : null;
    return key !== null && options.some((option) => option.value === key) ? [key] : [];
  }, [options, value]);
  const selectIds = useMemo(() => (id ? { trigger: `${id}-select` } : undefined), [id]);
  const onSelectValueChange = useCallback(
    ({ value: next }: { value: string[] }) => {
      const nextValue = next[0];

      if (nextValue !== undefined) {
        onChange(nextValue);
      }
    },
    [onChange]
  );

  return (
    <Select
      aria-label={title}
      className="nodrag"
      collection={collection}
      ids={selectIds}
      invalid={invalid}
      size="xs"
      value={selectedValue}
      valueTextProps={SELECT_VALUE_TEXT_PROPS}
      w="full"
      onValueChange={onSelectValueChange}
    />
  );
};

const EnumInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const options = useMemo(
    () =>
      (template.options ?? []).map((option) => ({
        label: template.uiChoiceLabels?.[String(option)] ?? String(option),
        value: String(option),
      })),
    [template.options, template.uiChoiceLabels]
  );
  const onOptionChange = useCallback(
    (nextValue: string) => {
      const option = template.options?.find((candidate) => String(candidate) === nextValue);
      onChange(option ?? nextValue);
    },
    [onChange, template.options]
  );

  if (template.name === 'scheduler') {
    return (
      <Combobox
        aria-label={template.title}
        className="nodrag nowheel"
        id={id ? `${id}-scheduler-combobox` : undefined}
        invalid={invalid}
        options={options}
        size="xs"
        value={typeof value === 'string' ? value : null}
        onValueChange={onOptionChange}
      />
    );
  }

  return (
    <SelectInput
      id={id}
      invalid={invalid}
      options={options}
      title={template.title}
      value={value}
      onChange={onOptionChange}
    />
  );
};

const DEFAULT_MODEL_TYPES: ModelTaxonomyType[] = ['main', 'vae', 'lora', 'controlnet', 't2i_adapter', 'ip_adapter'];

const ModelIdentifierInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const selectedKey =
    typeof (value as { key?: unknown } | null)?.key === 'string' ? (value as { key: string }).key : null;
  const modelTypes = (template.uiModelType ?? DEFAULT_MODEL_TYPES) as ModelTaxonomyType[];
  const allowedBases = template.uiModelBase;
  // Filter by model format as well as base/type so loaders cannot receive unsupported component layouts.
  const allowedFormats = template.uiModelFormat;
  const filter = useCallback(
    (model: ModelConfig) =>
      (allowedBases ? allowedBases.includes(model.base) : true) &&
      // A components-only folder carries no transformer, so it can only fill a field that asks for
      // folders explicitly (e.g. a loader's Components field), never a format-agnostic model field.
      (allowedFormats ? allowedFormats.includes(model.format) : model.components_only !== true),
    [allowedBases, allowedFormats]
  );
  const onModelChange = useCallback(
    (model: ModelConfig | null) =>
      onChange(
        model ? { base: model.base, hash: model.hash, key: model.key, name: model.name, type: model.type } : undefined
      ),
    [onChange]
  );

  return (
    <Suspense fallback={MODEL_SELECT_FALLBACK}>
      <ModelSelect
        className="nodrag nowheel"
        filter={filter}
        id={id ? `${id}-model-combobox` : undefined}
        invalid={invalid}
        isClearable={false}
        modelTypes={modelTypes}
        size="xs"
        value={selectedKey}
        onChange={onModelChange}
      />
    </Suspense>
  );
};

const SchedulerInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => (
  <Combobox
    aria-label={template.title}
    className="nodrag nowheel"
    id={id ? `${id}-scheduler-combobox` : undefined}
    invalid={invalid}
    options={SCHEDULER_OPTIONS}
    size="xs"
    value={typeof value === 'string' ? value : null}
    onValueChange={onChange}
  />
);

let boardOptionsRequest: { owner: AccountScope; promise: Promise<GalleryBoard[]> } | null = null;

registerAccountOwnedResource({
  clear: () => {
    boardOptionsRequest = null;
  },
  name: 'workflow-board-options',
});

const getBoardOptions = (): Promise<GalleryBoard[]> => {
  const owner = captureAccountScope();

  if (boardOptionsRequest?.owner !== owner) {
    const promise = galleryDestinations
      .list({ signal: owner.signal })
      .then((loadedBoards) => {
        assertAccountScopeCurrent(owner);

        return loadedBoards.filter((board) => board.kind === 'board');
      })
      .catch((error: unknown) => {
        if (boardOptionsRequest?.promise === promise) {
          boardOptionsRequest = null;
        }
        throw error;
      });

    boardOptionsRequest = { owner, promise };
  }

  return boardOptionsRequest.promise;
};

const BoardInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const [boards, setBoards] = useState<GalleryBoard[]>([]);

  useEffect(() => {
    let isCancelled = false;

    getBoardOptions()
      .then((loadedBoards) => {
        if (!isCancelled) {
          setBoards(loadedBoards);
        }
      })
      .catch(() => {
        // Board listing is a convenience; the auto/none sentinels still work.
      });

    return () => {
      isCancelled = true;
    };
  }, []);

  const selected =
    value === 'auto' || value === 'none'
      ? value
      : typeof (value as { board_id?: unknown } | null)?.board_id === 'string'
        ? (value as { board_id: string }).board_id
        : 'auto';
  const options = useMemo(
    () => [
      { label: 'Auto', value: 'auto' },
      { label: 'None', value: 'none' },
      ...boards.map((board) => ({ label: board.name, value: board.id })),
    ],
    [boards]
  );
  const onBoardChange = useCallback(
    (next: string) => onChange(next === 'auto' || next === 'none' ? next : { board_id: next }),
    [onChange]
  );

  return (
    <SelectInput
      id={id}
      invalid={invalid}
      options={options}
      title={template.title}
      value={selected}
      onChange={onBoardChange}
    />
  );
};

const MEDIA_FIELD_CONFIG = {
  image: {
    fileAccept: 'image/*',
    getThumbnailUrl: (name: string) => galleryImageUrls.thumbnail(name),
    nameKey: 'image_name',
    noun: 'image',
  },
  video: {
    fileAccept: 'video/*,audio/*',
    getThumbnailUrl: (name: string) => galleryVideoUrls.thumbnail(name),
    nameKey: 'video_name',
    noun: 'video',
  },
} as const satisfies Record<
  WorkflowMediaKind,
  { fileAccept: string; getThumbnailUrl: (name: string) => string; nameKey: string; noun: string }
>;

const HIDDEN_FILE_INPUT_STYLE = { display: 'none' } as const;
const IMAGE_ONLY = ['image'] as const;
const VIDEO_ONLY = ['video'] as const;
const MEDIA_INPUT_FOCUS_PROPS = { outline: '2px solid {colors.accent.focusRing}', outlineOffset: '2px' } as const;

/**
 * Apply uploads only to their originating mounted widget/project. Late successes still refresh Gallery and direct
 * users there instead of mutating another project.
 */
const useMediaUpload = ({
  kind,
  multiple = false,
  onUploaded,
}: {
  kind: WorkflowMediaKind;
  multiple?: boolean;
  onUploaded: (names: string[]) => void;
}) => {
  const config = MEDIA_FIELD_CONFIG[kind];
  const uploadBoardId = useWorkflowProjectSelector((project) =>
    typeof project.galleryValues.selectedBoardId === 'string' ? project.galleryValues.selectedBoardId : 'none'
  );
  const { project } = useWorkflowUi();
  const queryClient = useQueryClient();
  const isMountedRef = useRef(true);

  useMountEffect(() => () => {
    isMountedRef.current = false;
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const onUploadClick = useCallback(() => fileInputRef.current?.click(), []);
  const onFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.currentTarget.files ?? []).slice(0, multiple ? undefined : 1);

      // Reset so picking the same file again re-fires the change event.
      event.currentTarget.value = '';

      if (files.length === 0) {
        return;
      }

      // `accept` on the input is advisory only ("All Files" bypasses it); an
      // unknown type (empty string) is left for the server to judge.
      if (files.some((file) => file.type && !file.type.startsWith(`${kind}/`))) {
        toaster.create({
          title: `Please choose ${multiple ? `${config.noun} files` : `a ${config.noun} file`}`,
          type: 'error',
        });

        return;
      }

      const owner = captureAccountScope();
      const projectId = project.getSnapshot().id;

      setIsUploading(true);
      void (async () => {
        const names: string[] = [];
        const failedFiles: string[] = [];

        // Each file lands or fails on its own; what landed is adopted either way.
        for (const file of files) {
          try {
            names.push(
              kind === 'image'
                ? (await galleryTransfers.upload(file, uploadBoardId, { signal: owner.signal })).imageName
                : (await galleryTransfers.uploadVideo(file, uploadBoardId, { signal: owner.signal })).name
            );
          } catch {
            if (!isAccountScopeCurrent(owner)) {
              return;
            }

            failedFiles.push(file.name);
          }
        }

        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        if (names.length > 0) {
          void invalidateGallery(queryClient, owner);

          if (isMountedRef.current && project.getSnapshot().id === projectId) {
            onUploaded(names);
          } else {
            toaster.create({
              description: `The workflow changed while ${names.join(', ')} uploaded - find it in the gallery.`,
              title: 'Upload finished',
              type: 'info',
            });
          }
        }

        if (failedFiles.length > 0) {
          toaster.create({
            description: failedFiles.join(', '),
            title: `Failed to upload ${failedFiles.length === 1 ? config.noun : `${failedFiles.length} ${config.noun}s`}`,
            type: 'error',
          });
        }

        if (isMountedRef.current) {
          setIsUploading(false);
        }
      })();
    },
    [config.noun, kind, multiple, onUploaded, project, queryClient, uploadBoardId]
  );

  return { fileInputRef, isUploading, onFileChange, onUploadClick };
};

const getImageCollectionNames = (value: unknown): string[] =>
  Array.isArray(value)
    ? [
        ...new Set(
          value.flatMap((item) =>
            typeof (item as Record<string, unknown> | null)?.image_name === 'string'
              ? [(item as { image_name: string }).image_name]
              : []
          )
        ),
      ]
    : [];

const IMAGE_COLLECTION_REMOVE_PROPS = { opacity: 1 } as const;

const ImageCollectionTile = ({
  index,
  name,
  onRemove,
}: {
  index: number;
  name: string;
  onRemove: (index: number) => void;
}) => {
  const onRemoveClick = useCallback(() => onRemove(index), [index, onRemove]);

  return (
    <Box aspectRatio="1" bg="bg.subtle" className="group" position="relative" rounded="xs">
      <Image
        alt=""
        h="full"
        objectFit="cover"
        rounded="xs"
        src={MEDIA_FIELD_CONFIG.image.getThumbnailUrl(name)}
        title={name}
        w="full"
      />
      <IconButton
        aria-label={`Remove ${name}`}
        bg="bg.panel"
        insetInlineEnd="0.5"
        opacity={0}
        position="absolute"
        size="2xs"
        top="0.5"
        transition="opacity var(--wb-motion-duration-fast) ease"
        variant="subtle"
        _focusVisible={IMAGE_COLLECTION_REMOVE_PROPS}
        _groupHover={IMAGE_COLLECTION_REMOVE_PROPS}
        onClick={onRemoveClick}
      >
        <Icon as={XIcon} boxSize="3" />
      </IconButton>
    </Box>
  );
};

const ImageCollectionDropMonitor = ({ dropId, onDrop }: { dropId: string; onDrop: (names: string[]) => void }) => {
  const onDragEnd = useCallback(
    (event: DragEndEvent) => {
      if (event.over?.id === dropId) {
        onDrop(getWorkflowMediaFieldDropItems(event.active.data.current, 'image').map((item) => item.name));
      }
    },
    [dropId, onDrop]
  );

  useDndMonitor({ onDragEnd });

  return null;
};

const MediaDropMonitor = ({
  dropId,
  kind,
  onDrop,
}: {
  dropId: string;
  kind: WorkflowMediaKind;
  onDrop: (item: { name: string }) => void;
}) => {
  const onDragEnd = useCallback(
    (event: DragEndEvent) => {
      if (event.over?.id !== dropId) {
        return;
      }

      const item = getWorkflowMediaFieldDropItem(event.active.data.current, kind);

      if (item) {
        onDrop(item);
      }
    },
    [dropId, kind, onDrop]
  );

  useDndMonitor({ onDragEnd });

  return null;
};

const ImageCollectionInput = ({ id, invalid, nodeId, onChange, template, value }: WorkflowFieldInputProps) => {
  const { t } = useTranslation();
  const { project } = useWorkflowUi();
  const names = useMemo(() => getImageCollectionNames(value), [value]);
  const invalidAriaProps = useMemo(() => (invalid ? { 'aria-invalid': true } : {}), [invalid]);
  // An upload finishing later appends to the field as it stands then, read from the document, not
  // to the list this render saw.
  const appendNames = useCallback(
    (added: string[]) => {
      const node = project.getSnapshot().projectGraph.nodes.find((candidate) => candidate.id === nodeId);
      const current = getImageCollectionNames(
        node && isInvocationNode(node) ? node.data.inputs[template.name]?.value : value
      );
      const next = [...current, ...added.filter((name) => !current.includes(name))];

      if (next.length !== current.length) {
        onChange(next.map((image_name) => ({ image_name })));
      }
    },
    [nodeId, onChange, project, template.name, value]
  );
  const removeAt = useCallback(
    (index: number) => onChange(names.filter((_, i) => i !== index).map((image_name) => ({ image_name }))),
    [names, onChange]
  );
  const onClearClick = useCallback(() => onChange([]), [onChange]);

  const instanceId = useId();
  const dropId = getWorkflowMediaFieldDropId(`${id ?? 'field'}:${instanceId}`);
  const { active } = useDndContext();
  const acceptsActiveDrag = getWorkflowMediaFieldDropItems(active?.data.current, 'image').length > 0;
  const { isOver, setNodeRef } = useDroppable({ disabled: !acceptsActiveDrag, id: dropId });
  const { fileInputRef, isUploading, onFileChange, onUploadClick } = useMediaUpload({
    kind: 'image',
    multiple: true,
    onUploaded: appendNames,
  });
  const pickerSelection = useMemo<GalleryPickerSelection>(
    () => ({
      addedKeys: new Set(names.map((name) => toGalleryItemKey({ kind: 'image', name }))),
      mode: 'multiple',
      remaining: {},
    }),
    [names]
  );
  const onPick = useCallback((item: GalleryItem) => appendNames([item.name]), [appendNames]);
  const pickerLabel = t('widgets.gallery.picker.chooseImage');

  return (
    <Box position="relative" w="full" {...invalidAriaProps}>
      <ImageCollectionDropMonitor dropId={dropId} onDrop={appendNames} />
      <Box
        ref={setNodeRef}
        boxShadow={invalid ? '0 0 0 1px {colors.red.solid}' : undefined}
        className="nodrag"
        position="relative"
        rounded="sm"
        w="full"
      >
        {names.length > 0 ? (
          <SimpleGrid
            borderWidth="1px"
            className="nowheel"
            columns={3}
            gap="1"
            maxH="48"
            overflowY="auto"
            p="1"
            rounded="sm"
          >
            {names.map((name, index) => (
              <ImageCollectionTile key={name} index={index} name={name} onRemove={removeAt} />
            ))}
          </SimpleGrid>
        ) : (
          // The empty state keeps the drop target tall enough to hit and doubles as the picker trigger.
          <GalleryPickerPopover accept={IMAGE_ONLY} label={pickerLabel} selection={pickerSelection} onPick={onPick}>
            <chakra.button
              aria-label={pickerLabel}
              className="nodrag"
              display="block"
              h="20"
              type="button"
              w="full"
              _focusVisible={MEDIA_INPUT_FOCUS_PROPS}
            >
              <Flex
                alignItems="center"
                borderStyle="dashed"
                borderWidth="1px"
                direction="column"
                gap="1"
                h="full"
                justifyContent="center"
                rounded="sm"
                transition="border-color var(--wb-motion-duration-fast) ease"
                w="full"
                _hover={MEDIA_INPUT_HOVER_PROPS}
              >
                <Text as="span" color="fg" fontSize="xs" fontWeight="600">
                  {pickerLabel}
                </Text>
                <Text as="span" color="fg.subtle" fontSize="2xs">
                  {t('widgets.gallery.picker.dropHint')}
                </Text>
              </Flex>
            </chakra.button>
          </GalleryPickerPopover>
        )}
        <DropTargetOverlay isActive={acceptsActiveDrag} isOver={isOver} label="Drop images" />
      </Box>
      <HStack gap="1.5" mt="1" w="full">
        <GalleryPickerPopover accept={IMAGE_ONLY} label={pickerLabel} selection={pickerSelection} onPick={onPick}>
          <Button className="nodrag" size="2xs" variant="outline">
            <Icon as={ImagePlusIcon} boxSize="3" />
            {t('common.add')}
          </Button>
        </GalleryPickerPopover>
        <Button className="nodrag" disabled={isUploading} size="2xs" variant="outline" onClick={onUploadClick}>
          {isUploading ? 'Uploading…' : 'Upload'}
        </Button>
        {names.length > 0 ? (
          <Button className="nodrag" size="2xs" variant="ghost" onClick={onClearClick}>
            Clear
          </Button>
        ) : null}
        {names.length > 0 ? (
          <Text color="fg.subtle" fontSize="2xs" ms="auto">
            {t('nodes.imageCollectionCount', { count: names.length })}
          </Text>
        ) : null}
        <input
          ref={fileInputRef}
          accept={MEDIA_FIELD_CONFIG.image.fileAccept}
          aria-label="Upload image files"
          multiple
          style={HIDDEN_FILE_INPUT_STYLE}
          type="file"
          onChange={onFileChange}
        />
      </HStack>
    </Box>
  );
};

const MediaInput = ({ id, invalid, kind, onChange, value }: WorkflowFieldInputProps & { kind: WorkflowMediaKind }) => {
  const { t } = useTranslation();
  const config = MEDIA_FIELD_CONFIG[kind];
  const mediaName =
    typeof (value as Record<string, unknown> | null | undefined)?.[config.nameKey] === 'string'
      ? ((value as Record<string, string>)[config.nameKey] ?? null)
      : null;
  const invalidAriaProps = useMemo(() => (invalid ? { 'aria-invalid': true } : {}), [invalid]);

  // Use instance-unique drop IDs because editor and Linear UI can render the same field simultaneously.
  const instanceId = useId();
  const dropId = getWorkflowMediaFieldDropId(`${id ?? 'field'}:${instanceId}`);
  const { active } = useDndContext();
  const acceptsActiveDrag = getWorkflowMediaFieldDropItem(active?.data.current, kind) !== null;
  const { isOver, setNodeRef } = useDroppable({ disabled: !acceptsActiveDrag, id: dropId });
  const onUploaded = useCallback(
    (names: string[]) => onChange({ [config.nameKey]: names[0] }),
    [config.nameKey, onChange]
  );
  const { fileInputRef, isUploading, onFileChange, onUploadClick } = useMediaUpload({ kind, onUploaded });

  const onPick = useCallback(
    (item: GalleryItem) => onChange({ [config.nameKey]: item.name }),
    [config.nameKey, onChange]
  );
  const onClearClick = useCallback(() => onChange(undefined), [onChange]);

  // Replace failed stale thumbnails with media icons; retry when the value changes.
  const [failedThumbnail, setFailedThumbnail] = useState<string | null>(null);
  const onThumbnailError = useCallback(() => setFailedThumbnail(mediaName), [mediaName]);

  // Resolve badge metadata best-effort; name-based preview remains usable if lookup fails.
  const { data: mediaItem } = useQuery({
    enabled: mediaName !== null && mediaName !== '',
    queryFn: ({ signal }) => galleryItems.resolve({ kind, name: mediaName ?? '' }, signal),
    queryKey: ['workflow-media-field-item', kind, mediaName],
    retry: false,
    staleTime: 60_000,
  });
  const badge =
    mediaItem && mediaItem.name === mediaName && mediaItem.width > 0
      ? `${mediaItem.width}x${mediaItem.height}${
          mediaItem.kind === 'video' ? ` · ${formatGalleryVideoDuration(mediaItem.durationSeconds)}` : ''
        }`
      : null;

  const FallbackIcon = kind === 'video' ? FilmIcon : ImageIcon;
  const pickerAccept = kind === 'video' ? VIDEO_ONLY : IMAGE_ONLY;
  const pickerLabel = t(kind === 'video' ? 'widgets.gallery.picker.chooseVideo' : 'widgets.gallery.picker.chooseImage');
  const onMediaDrop = useCallback(
    (item: { name: string }) => onChange({ [config.nameKey]: item.name }),
    [config.nameKey, onChange]
  );

  return (
    <Box position="relative" w="full" {...invalidAriaProps}>
      <MediaDropMonitor dropId={dropId} kind={kind} onDrop={onMediaDrop} />
      {/* The whole preview area is the drop target, like the legacy editor's widget. */}
      <Box
        ref={setNodeRef}
        boxShadow={invalid ? '0 0 0 1px {colors.red.solid}' : undefined}
        className="nodrag"
        h="32"
        position="relative"
        rounded="sm"
        w="full"
      >
        <GalleryPickerPopover accept={pickerAccept} label={pickerLabel} onPick={onPick}>
          <chakra.button
            aria-label={pickerLabel}
            className="nodrag"
            display="block"
            h="full"
            type="button"
            w="full"
            _focusVisible={MEDIA_INPUT_FOCUS_PROPS}
          >
            {mediaName ? (
              <Flex
                alignItems="center"
                borderWidth="1px"
                h="full"
                justifyContent="center"
                overflow="hidden"
                rounded="sm"
                transition="border-color var(--wb-motion-duration-fast) ease"
                w="full"
                _hover={MEDIA_INPUT_HOVER_PROPS}
              >
                {failedThumbnail !== mediaName ? (
                  <Image
                    alt=""
                    maxH="full"
                    maxW="full"
                    objectFit="contain"
                    src={config.getThumbnailUrl(mediaName)}
                    title={mediaName}
                    onError={onThumbnailError}
                  />
                ) : (
                  <Box title={mediaName}>
                    <Icon as={FallbackIcon} boxSize="10" color="fg.subtle" />
                  </Box>
                )}
              </Flex>
            ) : (
              <Flex
                alignItems="center"
                borderStyle="dashed"
                borderWidth="1px"
                direction="column"
                gap="1"
                h="full"
                justifyContent="center"
                rounded="sm"
                transition="border-color var(--wb-motion-duration-fast) ease"
                w="full"
                _hover={MEDIA_INPUT_HOVER_PROPS}
              >
                <Text as="span" color="fg" fontSize="xs" fontWeight="600">
                  {pickerLabel}
                </Text>
                <Text as="span" color="fg.subtle" fontSize="2xs">
                  {t('widgets.gallery.picker.dropHint')}
                </Text>
              </Flex>
            )}
          </chakra.button>
        </GalleryPickerPopover>
        {mediaName && badge !== null ? (
          <Badge
            bottom="1"
            fontVariantNumeric="tabular-nums"
            insetInlineEnd="1"
            pointerEvents="none"
            position="absolute"
            size="xs"
            variant="solid"
          >
            {badge}
          </Badge>
        ) : null}
        <DropTargetOverlay isActive={acceptsActiveDrag} isOver={isOver} label={`Drop ${config.noun}`} />
      </Box>
      <HStack gap="1.5" mt="1" w="full">
        <Button
          className="nodrag"
          disabled={isUploading}
          size="2xs"
          title={`Upload a ${config.noun} and use it here`}
          variant="outline"
          onClick={onUploadClick}
        >
          {isUploading ? 'Uploading…' : 'Upload'}
        </Button>
        {mediaName ? (
          <Button className="nodrag" size="2xs" variant="ghost" onClick={onClearClick}>
            Clear
          </Button>
        ) : null}
        <input
          ref={fileInputRef}
          accept={config.fileAccept}
          aria-label={`Upload ${config.noun} file`}
          style={HIDDEN_FILE_INPUT_STYLE}
          type="file"
          onChange={onFileChange}
        />
      </HStack>
    </Box>
  );
};

/**
 * Convention shared with the legacy editor: a `video-frame-index` field's
 * source video lives on a sibling field named `video` on the same node.
 */
const COMPANION_VIDEO_FIELD_NAME = 'video';

/**
 * Share one frame value across input, scrubber, and native video preview. Fall back to plain input when the
 * companion video/rate cannot resolve.
 */
const VideoFrameIndexInput = (props: WorkflowFieldInputProps) => {
  const { nodeId, onChange, value } = props;

  // A connected `video` field keeps its stored value in the document, so the
  // sibling value alone would preview a stale video while invoke uses the
  // upstream one. Resolved edges are the same "connected" signal the editor
  // and Linear panel use to hide the video widget itself.
  const isVideoConnected = useWorkflowProjectSelector(
    (project) =>
      nodeId !== undefined &&
      getResolvedWorkflowEdges(project.projectGraph.nodes, project.projectGraph.edges).some(
        (edge) => edge.target === nodeId && edge.targetHandle === COMPANION_VIDEO_FIELD_NAME
      )
  );
  const siblingVideoName = useWorkflowProjectSelector((project) => {
    const node = nodeId ? project.projectGraph.nodes.find((candidate) => candidate.id === nodeId) : undefined;

    if (!node || !isInvocationNode(node)) {
      return null;
    }

    const sibling = node.data.inputs[COMPANION_VIDEO_FIELD_NAME]?.value as Record<string, unknown> | undefined;

    return typeof sibling?.video_name === 'string' && sibling.video_name !== '' ? sibling.video_name : null;
  });
  const videoName = isVideoConnected ? null : siblingVideoName;

  // Same query key as the media widget's badge lookup, so the two share a cache
  // entry when a frame field sits next to a populated video field (the common case).
  const { data, isError } = useQuery({
    enabled: videoName !== null,
    queryFn: ({ signal }) => galleryItems.resolve({ kind: 'video', name: videoName ?? '' }, signal),
    queryKey: ['workflow-media-field-item', 'video', videoName],
    retry: false,
    staleTime: 60_000,
  });
  // Rebind: useQuery's NoInfer-wrapped `data` defeats discriminant narrowing.
  const videoItem: GalleryItem | undefined = data;
  const video = videoItem && videoItem.kind === 'video' && videoItem.name === videoName ? videoItem : null;

  // Use estimated frame count only for scrubbing bounds; backend invocation resolves authoritative decoder
  // indices.
  const fps = video?.fps ?? null;
  const frameCount =
    video && fps && video.durationSeconds > 0 ? Math.max(1, Math.round(video.durationSeconds * fps)) : null;

  // Resolve negative indices (-1 = last frame) for display only — the field
  // value is preserved verbatim so users can still type "-1" and have the
  // backend resolve it against the actual frame count.
  const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  const resolvedIndex =
    frameCount === null
      ? 0
      : Math.max(0, Math.min(frameCount - 1, numericValue < 0 ? frameCount + numericValue : numericValue));

  return (
    <Flex direction="column" gap="1" w="full">
      <NumericInput {...props} />
      {video && fps !== null && frameCount !== null ? (
        <FrameScrubber
          fps={fps}
          frameCount={frameCount}
          resolvedIndex={resolvedIndex}
          videoUrl={video.fullUrl}
          onChange={onChange}
        />
      ) : (
        <Flex borderStyle="dashed" borderWidth="1px" justifyContent="center" px="2" py="2" rounded="sm">
          <Text color="fg.subtle" fontSize="xs" textAlign="center">
            {isVideoConnected
              ? 'Frame preview unavailable while the video comes from a graph connection.'
              : videoName === null
                ? "Set this node's Video field to preview frames."
                : isError
                  ? 'Frame preview unavailable — the video could not be loaded (it may have been deleted).'
                  : video === null
                    ? 'Loading frame preview…'
                    : 'Frame preview unavailable — the video has no probed frame rate.'}
          </Text>
        </Flex>
      )}
    </Flex>
  );
};

const FRAME_SLIDER_ARIA_LABEL = ['Frame'];
const FRAME_VIDEO_STYLE = { height: '100%', objectFit: 'contain', width: '100%' } as const;

const FrameScrubber = ({
  fps,
  frameCount,
  onChange,
  resolvedIndex,
  videoUrl,
}: {
  fps: number;
  frameCount: number;
  onChange: (value: unknown) => void;
  resolvedIndex: number;
  videoUrl: string;
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Seek inside each frame's display interval to avoid boundary decoding artifacts.
  useEffect(() => {
    const el = videoRef.current;

    if (!el) {
      return;
    }

    const setTime = () => {
      try {
        el.currentTime = (resolvedIndex + 0.5) / fps;
      } catch {
        // Some browsers throw when seeking an element that isn't ready; the
        // preview keeps its previous frame and the next effect run re-seeks.
      }
    };

    if (el.readyState >= 1) {
      setTime();
    } else {
      el.addEventListener('loadedmetadata', setTime, { once: true });

      return () => el.removeEventListener('loadedmetadata', setTime);
    }
  }, [fps, resolvedIndex, videoUrl]);

  const onSliderChange = useCallback(
    ({ value: next }: { value: number[] }) => {
      if (next[0] !== undefined) {
        onChange(next[0]);
      }
    },
    [onChange]
  );
  const sliderValue = useMemo(() => [resolvedIndex], [resolvedIndex]);

  return (
    <Flex className="nodrag" direction="column" gap="1" w="full">
      <Box borderWidth="1px" h="32" overflow="hidden" position="relative" rounded="sm" w="full">
        <video ref={videoRef} muted playsInline preload="auto" src={videoUrl} style={FRAME_VIDEO_STYLE} />
        <Badge
          bottom="1"
          fontVariantNumeric="tabular-nums"
          insetInlineEnd="1"
          pointerEvents="none"
          position="absolute"
          size="xs"
          variant="solid"
        >
          {`${resolvedIndex} / ${frameCount - 1}`}
        </Badge>
      </Box>
      {/* A single-frame video would give the slider min === max, which zag renders as NaN% CSS. */}
      {frameCount > 1 ? (
        <Slider
          aria-label={FRAME_SLIDER_ARIA_LABEL}
          max={frameCount - 1}
          min={0}
          size="sm"
          step={1}
          value={sliderValue}
          withThumbTooltip
          onValueChange={onSliderChange}
        />
      ) : null}
    </Flex>
  );
};

/** Convert workflow alpha integers 0–255 locally; platform RgbaColor uses unit alpha. */
const toColorFieldValue = (color: string): Record<string, number> => {
  const { a, b, g, r } = parseHexColor(color);

  return { a: Math.round(a * 255), b, g, r };
};

const fromColorFieldValue = (value: unknown): string => {
  const channels = (typeof value === 'object' && value !== null ? value : {}) as Partial<Record<string, number>>;

  return formatHexColor(
    {
      a: (channels.a ?? 255) / 255,
      b: channels.b ?? 0,
      g: channels.g ?? 0,
      r: channels.r ?? 0,
    },
    { alpha: true }
  );
};

const ColorInput = ({ invalid, onChange, value }: WorkflowFieldInputProps) => {
  const color = fromColorFieldValue(value);
  const handleChange = useCallback((next: string) => onChange(toColorFieldValue(next)), [onChange]);

  return (
    // `nodrag` keeps a click on the swatch from panning the node canvas.
    <HStack className="nodrag" gap="2" w="full" {...invalidProps(invalid)}>
      <ColorPicker aria-label="Color" value={color} withAlpha withValueText onValueChange={handleChange} />
    </HStack>
  );
};

const LORA_MODEL_TYPES: ModelTaxonomyType[] = ['lora'];

/**
 * Edit LoRA collections by index so imported duplicate keys and unreadable entries remain independent and
 * preserved.
 */
const LoRACollectionInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  // The raw list, unreadable items included — see `toLoraFieldCollectionList`.
  const entries = useMemo(() => toLoraFieldCollectionList(value), [value]);
  const selectedKeys = useMemo(
    () => new Set(entries.filter(isLoraFieldCollectionEntry).map((entry) => entry.lora.key)),
    [entries]
  );
  // The loaders stamp `ui_model_base`, so each node offers only the LoRAs it can actually apply.
  const allowedBases = template.uiModelBase;
  const filter = useCallback(
    (model: ModelConfig) => (allowedBases ? allowedBases.includes(model.base) : true),
    [allowedBases]
  );
  // Write emptied collections as undefined to restore the loaders' actual default.
  const commit = useCallback((next: unknown[]) => onChange(next.length === 0 ? undefined : next), [onChange]);
  const onAdd = useCallback(
    (model: ModelConfig | null) => {
      // The picker already excludes what is in the list; the guard covers a stale selection.
      if (!model || selectedKeys.has(model.key)) {
        return;
      }

      const defaultWeight = model.default_settings?.weight;
      const entry: LoraFieldCollectionEntry = {
        lora: { base: model.base, hash: model.hash, key: model.key, name: model.name, type: model.type },
        weight:
          typeof defaultWeight === 'number' && Number.isFinite(defaultWeight)
            ? defaultWeight
            : DEFAULT_LORA_WEIGHT_CONFIG.initial,
      };

      commit([...entries, entry]);
    },
    [commit, entries, selectedKeys]
  );
  const onRemove = useCallback(
    (index: number) => commit(entries.filter((_, entryIndex) => entryIndex !== index)),
    [commit, entries]
  );
  const onWeightChange = useCallback(
    (index: number, weight: number | null) =>
      commit(
        entries.map((entry, entryIndex) =>
          entryIndex === index && isLoraFieldCollectionEntry(entry) ? { ...entry, weight } : entry
        )
      ),
    [commit, entries]
  );

  return (
    <Stack gap="1" w="full">
      <Suspense fallback={MODEL_SELECT_FALLBACK}>
        <ModelSelect
          className="nodrag nowheel"
          excludeKeys={selectedKeys}
          filter={allowedBases ? filter : undefined}
          id={id ? `${id}-lora-combobox` : undefined}
          invalid={invalid}
          modelTypes={(template.uiModelType as ModelTaxonomyType[] | null) ?? LORA_MODEL_TYPES}
          placeholder="Add LoRA…"
          size="xs"
          value={null}
          onChange={onAdd}
        />
      </Suspense>
      {entries.length > 0 ? (
        <Stack className="nowheel" gap="1" maxH="40" overflowY="auto" w="full">
          {entries.map((item, index) => (
            <LoRACollectionRow
              key={isLoraFieldCollectionEntry(item) ? `${item.lora.key}-${index}` : `unreadable-${index}`}
              entry={isLoraFieldCollectionEntry(item) ? item : null}
              id={id}
              index={index}
              onRemove={onRemove}
              onWeightChange={onWeightChange}
            />
          ))}
        </Stack>
      ) : null}
    </Stack>
  );
};

/** The weight column edits a bounded float; the range is the same one the Generate LoRA controls use. */
const LORA_WEIGHT_TEMPLATE: FieldInputTemplate = {
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'direct',
  maximum: LORA_FIELD_WEIGHT_RANGE.max,
  minimum: LORA_FIELD_WEIGHT_RANGE.min,
  multipleOf: null,
  name: 'weight',
  options: null,
  required: true,
  title: 'Weight',
  type: { batch: false, cardinality: 'SINGLE', name: 'FloatField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
};

const LoRACollectionRow = ({
  entry,
  id,
  index,
  onRemove,
  onWeightChange,
}: {
  entry: LoraFieldCollectionEntry | null;
  id?: string;
  index: number;
  onRemove: (index: number) => void;
  onWeightChange: (index: number, weight: number | null) => void;
}) => {
  const label = entry ? entry.lora.name : 'Unreadable entry';
  const onRemoveClick = useCallback(() => onRemove(index), [index, onRemove]);
  // The shared numeric control commits as typed; a cleared or out-of-range weight stays on screen and blocks
  // invoking through the field's own reason instead of being clamped.
  const onValueChange = useCallback(
    (weight: unknown) => onWeightChange(index, typeof weight === 'number' ? weight : null),
    [index, onWeightChange]
  );

  return (
    <HStack gap="1" minW="0" w="full">
      <Tooltip content={entry ? label : 'This entry is not a readable LoRA and will block invoking.'}>
        <MiddleTruncate color={entry ? undefined : 'fg.error'} flex="1" fontSize="2xs" minW="0" text={label} />
      </Tooltip>
      {entry ? (
        <Field.Root flexShrink="0" invalid={!isLoraFieldWeightValid(entry.weight)} w="16">
          <NumericInput
            ariaLabel={`${label} weight`}
            id={id ? `${id}-lora-${index}` : undefined}
            invalid={!isLoraFieldWeightValid(entry.weight)}
            step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
            template={LORA_WEIGHT_TEMPLATE}
            value={entry.weight ?? undefined}
            onChange={onValueChange}
          />
        </Field.Root>
      ) : null}
      <IconButton
        aria-label={`Remove ${label}`}
        className="nodrag"
        color="fg.muted"
        flexShrink="0"
        size="2xs"
        variant="ghost"
        onClick={onRemoveClick}
      >
        <Trash2Icon />
      </IconButton>
    </HStack>
  );
};

const ScalarCollectionInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const { t } = useTranslation();
  const items = useMemo<readonly unknown[]>(() => (Array.isArray(value) ? value : []), [value]);
  const isString = template.type.name === 'StringField';
  // Each row edits one scalar on a single line, whatever the list's own ui_component says.
  const itemTemplate = useMemo<FieldInputTemplate>(
    () => ({ ...template, type: { ...template.type, cardinality: 'SINGLE' }, uiComponent: null }),
    [template]
  );
  // Emptying the list restores the template's own default, so the reset affordance stays quiet.
  const clearsToUndefined = template.default === undefined && !template.required;
  const commit = useCallback(
    (next: unknown[]) => onChange(next.length === 0 && clearsToUndefined ? undefined : next),
    [clearsToUndefined, onChange]
  );
  const onAdd = useCallback(
    () => commit([...items, isString ? '' : (finiteNumberOrUndefined(template.minimum) ?? 0)]),
    [commit, isString, items, template.minimum]
  );
  const addButtonRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  // Rows are keyed by position, so a removal only unmounts the last row's controls. Move keyboard focus off
  // anything about to unmount before the commit lands, or it falls to the canvas and its delete shortcut.
  const onClear = useCallback(() => {
    addButtonRef.current?.focus();
    commit([]);
  }, [commit]);
  const onRemove = useCallback(
    (index: number) => {
      if (index === items.length - 1) {
        const removeButtons = listRef.current?.querySelectorAll<HTMLButtonElement>('[data-collection-remove]');

        (index > 0 ? removeButtons?.[index - 1] : addButtonRef.current)?.focus();
      }

      commit(items.filter((_, itemIndex) => itemIndex !== index));
    },
    [commit, items]
  );
  // A cleared or unparseable number row is kept as null: the list keeps its shape and the row reports itself.
  const onItemChange = useCallback(
    (index: number, next: unknown) =>
      commit(items.map((item, itemIndex) => (itemIndex === index ? (next === undefined ? null : next) : item))),
    [commit, items]
  );

  return (
    <Stack gap="1" w="full">
      {items.length > 0 ? (
        <Stack
          ref={listRef}
          borderWidth="1px"
          boxShadow={invalid ? '0 0 0 1px {colors.red.solid}' : undefined}
          className="nowheel"
          gap="1"
          maxH="40"
          overflowY="auto"
          p="1"
          rounded="sm"
          w="full"
        >
          {items.map((item, index) => (
            <ScalarCollectionRow
              key={index}
              id={id}
              index={index}
              itemTemplate={itemTemplate}
              value={item}
              onItemChange={onItemChange}
              onRemove={onRemove}
            />
          ))}
        </Stack>
      ) : null}
      <HStack gap="1.5" w="full">
        <Button ref={addButtonRef} className="nodrag" size="2xs" variant="outline" onClick={onAdd}>
          <Icon as={PlusIcon} boxSize="3" />
          {t('nodes.addItem')}
        </Button>
        {items.length > 0 ? (
          <Button className="nodrag" size="2xs" variant="ghost" onClick={onClear}>
            {t('common.clear')}
          </Button>
        ) : null}
        {items.length > 0 ? (
          <Text color="fg.subtle" fontSize="2xs" ms="auto">
            {t('nodes.collectionItemCount', { count: items.length })}
          </Text>
        ) : null}
      </HStack>
    </Stack>
  );
};

const ScalarCollectionRow = ({
  id,
  index,
  itemTemplate,
  onItemChange,
  onRemove,
  value,
}: {
  id?: string;
  index: number;
  itemTemplate: FieldInputTemplate;
  onItemChange: (index: number, next: unknown) => void;
  onRemove: (index: number) => void;
  value: unknown;
}) => {
  const { t } = useTranslation();
  const onChange = useCallback((next: unknown) => onItemChange(index, next), [index, onItemChange]);
  const onRemoveClick = useCallback(() => onRemove(index), [index, onRemove]);
  const Control = itemTemplate.type.name === 'StringField' ? StringInput : NumericInput;
  const invalid = !isWorkflowCollectionItemValid(itemTemplate, value);
  const removeLabel = t('nodes.removeItem', { field: itemTemplate.title, index: index + 1 });

  return (
    <HStack gap="1" w="full">
      <Text color="fg.subtle" flexShrink="0" fontSize="2xs" fontVariantNumeric="tabular-nums" minW="4" textAlign="end">
        {index + 1}.
      </Text>
      {/* The host's Field.Root marks every control inside it invalid; a row scopes its own validity instead. */}
      <Field.Root flex="1" invalid={invalid} minW="0">
        <Control
          ariaLabel={t('nodes.collectionItemLabel', { field: itemTemplate.title, index: index + 1 })}
          id={id ? `${id}-item-${index}` : undefined}
          invalid={invalid}
          template={itemTemplate}
          value={value}
          onChange={onChange}
        />
      </Field.Root>
      <Tooltip content={removeLabel}>
        <IconButton
          aria-label={removeLabel}
          className="nodrag"
          color="fg.muted"
          data-collection-remove=""
          flexShrink="0"
          size="2xs"
          variant="ghost"
          onClick={onRemoveClick}
        >
          <Trash2Icon />
        </IconButton>
      </Tooltip>
    </HStack>
  );
};

const CONNECTION_ONLY_FALLBACK = (
  <Text color="fg.subtle" fontSize="2xs">
    Connection only
  </Text>
);

const SavedWorkflowInput = ({ nodeId, onChange, template, value }: WorkflowFieldInputProps) => {
  const { t } = useTranslation();
  const { commands } = useWorkflowUi();
  const [search, setSearch] = useState('');
  const deferredSearch = useDeferredValue(search);
  const ownedParams = useMemo(() => getSavedWorkflowPickerOwnedQuery(deferredSearch), [deferredSearch]);
  const sharedParams = useMemo(() => getSavedWorkflowPickerSharedQuery(deferredSearch), [deferredSearch]);
  const ownedQuery = useInfiniteQuery(savedWorkflowPickerQueryOptions(ownedParams));
  const sharedQuery = useInfiniteQuery(savedWorkflowPickerQueryOptions(sharedParams));
  const ownedItems = getWorkflowPagesItems(ownedQuery.data);
  const sharedItems = getWorkflowPagesItems(sharedQuery.data);
  const items = useMemo(() => mergeSavedWorkflowPickerItems(ownedItems, sharedItems), [ownedItems, sharedItems]);
  const workflowId = typeof value === 'string' ? value : '';
  const selectedInList = items.some((item) => item.workflow_id === workflowId);
  const detailQuery = useQuery({
    ...savedWorkflowDetailQueryOptions(workflowId),
    enabled: workflowId !== '' && !selectedInList,
  });
  const selectedWorkflow = detailQuery.data ? getSavedWorkflowListItemFromRecord(detailQuery.data) : undefined;
  const selectionState = useMemo(
    () => getSavedWorkflowSelectionState(items, workflowId, selectedWorkflow),
    [items, selectedWorkflow, workflowId]
  );
  const selectedOption = useMemo(() => {
    const option = getSavedWorkflowSelectionOption(selectionState);

    return option?.value === MISSING_WORKFLOW_OPTION_VALUE
      ? { ...option, label: t('nodes.savedWorkflowMissing') }
      : option;
  }, [selectionState, t]);
  const options = useMemo(() => {
    const base = buildSavedWorkflowOptions(items);

    if (selectedOption && !base.some((option) => option.value === selectedOption.value)) {
      return [selectedOption, ...base];
    }

    return base;
  }, [items, selectedOption]);
  const displayState = getSavedWorkflowDisplayState(selectionState);
  const clearSelection = useCallback(() => onChange(''), [onChange]);
  const onWorkflowChange = useCallback(
    (nextValue: string | null) => {
      if (nodeId && nextValue === workflowId && workflowId) {
        commands.editGraph({ nodeId, type: 'retryCallSavedWorkflow' });
        return;
      }

      onChange(nextValue);
    },
    [commands, nodeId, onChange, workflowId]
  );
  const retrySelection = useCallback(() => {
    if (nodeId && workflowId) {
      commands.editGraph({ nodeId, type: 'retryCallSavedWorkflow' });
    }
  }, [commands, nodeId, workflowId]);
  const fetchNextPage = useCallback(() => {
    if (shouldFetchNextSavedWorkflowPickerPage(ownedQuery)) {
      void ownedQuery.fetchNextPage();
    }

    if (shouldFetchNextSavedWorkflowPickerPage(sharedQuery)) {
      void sharedQuery.fetchNextPage();
    }
  }, [ownedQuery, sharedQuery]);
  const isLoading = ownedQuery.isLoading || sharedQuery.isLoading;
  const isFetching = ownedQuery.isFetching || sharedQuery.isFetching;
  const statusText =
    displayState.statusLabel === 'choose'
      ? t('nodes.savedWorkflowChoose')
      : displayState.statusLabel === 'missing'
        ? t('nodes.savedWorkflowMissing')
        : null;

  return (
    <Stack gap="1" minW="0" w="full">
      <HStack gap="1" minW="0" w="full">
        <Combobox
          aria-label={template.title}
          flex="1"
          noResultsText={t('nodes.noMatchingWorkflows')}
          options={options}
          searchPlaceholder={isLoading ? t('nodes.savedWorkflowListLoading') : t('nodes.savedWorkflowSearch')}
          value={selectedOption?.value ?? null}
          onInputValueChange={setSearch}
          onItemReselect={retrySelection}
          onListScrollToBottom={fetchNextPage}
          onValueChange={onWorkflowChange}
        />
        {nodeId && workflowId && detailQuery.isError ? (
          <IconButton
            aria-label={t('common.retry')}
            className="nodrag"
            size="xs"
            variant="ghost"
            onClick={retrySelection}
          >
            <RotateCcwIcon />
          </IconButton>
        ) : null}
        {workflowId ? (
          <IconButton
            aria-label={t('nodes.savedWorkflowClear')}
            className="nodrag"
            size="xs"
            variant="ghost"
            onClick={clearSelection}
          >
            <XIcon />
          </IconButton>
        ) : null}
      </HStack>
      {selectionState.status === 'selected' ? (
        <HStack flexWrap="wrap" gap="1" minW="0">
          <Text color="fg.muted" fontSize="2xs" minW="0" truncate>
            {selectionState.workflow.name}
          </Text>
          {displayState.badges.includes('unsupported') ? (
            <Badge fontSize="2xs">{t('nodes.savedWorkflowUnsupported')}</Badge>
          ) : null}
          {displayState.badges.includes('default') ? (
            <Badge fontSize="2xs">{t('nodes.savedWorkflowDefaultBadge')}</Badge>
          ) : null}
          {displayState.badges.includes('shared') ? (
            <Badge fontSize="2xs">{t('nodes.savedWorkflowShared')}</Badge>
          ) : null}
        </HStack>
      ) : (
        <Badge alignSelf="flex-start" fontSize="2xs">
          {statusText}
        </Badge>
      )}
      {displayState.compatibility?.message ? (
        <Text color="fg.subtle" fontSize="2xs">
          {displayState.compatibility.message}
        </Text>
      ) : null}
      {isFetching ? (
        <Text color="fg.subtle" fontSize="2xs">
          {t('nodes.savedWorkflowUpdating')}
        </Text>
      ) : null}
    </Stack>
  );
};

export const WorkflowFieldInput = (props: WorkflowFieldInputProps) => {
  // COLLECTION fields hold arrays; only image and scalar lists have a list widget. Other
  // collections stay connection-only even when a migrated linear-form element
  // points at them, since the single-value widget would write a bare value.
  if (props.template.type.cardinality === 'COLLECTION') {
    switch (props.template.type.name) {
      case 'ImageField':
        return <ImageCollectionInput {...props} />;
      case 'FloatField':
      case 'IntegerField':
      case 'StringField':
        return <ScalarCollectionInput {...props} />;
      default:
        return CONNECTION_ONLY_FALLBACK;
    }
  }

  if (isWorkflowGeneratorFieldTypeName(props.template.type.name)) {
    return (
      <Suspense fallback={RECORD_PICKER_FALLBACK}>
        <GeneratorFieldInput {...props} />
      </Suspense>
    );
  }

  switch (props.template.type.name) {
    case 'SavedWorkflowField':
      return <SavedWorkflowInput {...props} />;
    case 'StringField':
      return <StringInput {...props} />;
    case 'IntegerField':
      if (props.template.uiComponent === 'video-frame-index' && props.template.type.cardinality === 'SINGLE') {
        return <VideoFrameIndexInput {...props} />;
      }

      if (props.onSeedModeChange && isSeedInputField(props.template)) {
        return (
          <WorkflowSeedInput
            {...props}
            onSeedModeChange={props.onSeedModeChange}
            seedMode={props.seedMode ?? 'fixed'}
          />
        );
      }

      return <NumericInput {...props} />;
    case 'FloatField':
      return <NumericInput {...props} />;
    case 'BooleanField':
      return <BooleanInput {...props} />;
    case 'EnumField':
      return <EnumInput {...props} />;
    case 'LoRAField':
      return <LoRACollectionInput {...props} />;
    case 'ModelIdentifierField':
      return <ModelIdentifierInput {...props} />;
    case 'SchedulerField':
      return <SchedulerInput {...props} />;
    case 'StylePresetField':
      return (
        <Suspense fallback={RECORD_PICKER_FALLBACK}>
          <StylePresetInput {...props} />
        </Suspense>
      );
    case 'SystemPromptField':
      return (
        <Suspense fallback={RECORD_PICKER_FALLBACK}>
          <SystemPromptInput {...props} />
        </Suspense>
      );
    case 'BoardField':
      return <BoardInput {...props} />;
    case 'ImageField':
      return <MediaInput {...props} kind="image" />;
    case 'VideoField':
      return <MediaInput {...props} kind="video" />;
    case 'ColorField':
      return <ColorInput {...props} />;
    default:
      return CONNECTION_ONLY_FALLBACK;
  }
};
