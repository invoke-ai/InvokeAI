import type { ModelConfig, ModelTaxonomyType } from '@features/models/react';
import type { FieldInputTemplate } from '@features/workflow/contracts';
import type { LoraFieldCollectionEntry } from '@features/workflow/utility';

import {
  Badge,
  Box,
  createListCollection,
  Flex,
  HStack,
  Icon,
  Image,
  Input,
  NumberInput,
  Stack,
  Switch,
  Text,
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
import { getSelectedGalleryImageFromValues, getSelectedGalleryItemFromValues } from '@features/gallery/contracts';
import { invalidateGallery } from '@features/gallery/queries';
import { galleryImageUrls, galleryVideoUrls } from '@features/gallery/utility';
import { DEFAULT_LORA_WEIGHT_CONFIG, SCHEDULER_OPTIONS } from '@features/generation/settings';
import { isInvocationNode } from '@features/workflow/contracts';
import {
  getWorkflowMediaFieldDropId,
  getWorkflowMediaFieldDropItem,
  type WorkflowMediaKind,
} from '@features/workflow/ui/fields/mediaFieldDnd';
import { useWorkflowProjectSelector, useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import {
  getResolvedWorkflowEdges,
  isLoraFieldCollectionEntry,
  toLoraFieldCollectionList,
} from '@features/workflow/utility';
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
  DropZone,
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
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { FilmIcon, ImageIcon, Trash2Icon } from 'lucide-react';
import { lazy, Suspense, useCallback, useEffect, useId, useMemo, useRef, useState, type ChangeEvent } from 'react';

const ModelSelect = lazy(() => import('@features/models/react').then((module) => ({ default: module.ModelSelect })));
const MODEL_SELECT_FALLBACK = (
  <Button disabled size="xs" w="full">
    Loading models…
  </Button>
);

export const getWorkflowSelectedGalleryImage = getSelectedGalleryImageFromValues;
export const getWorkflowSelectedGalleryItem = getSelectedGalleryItemFromValues;

/**
 * Direct-input controls for workflow fields, shared between the node editor
 * and the Linear UI panel. Renders by template field type; connection-only
 * and unsupported types fall through to a muted note.
 */

export interface WorkflowFieldInputProps {
  id?: string;
  invalid?: boolean;
  /** Owning invocation node, when known — lets widgets read sibling fields (e.g. the frame scrubber's companion video). */
  nodeId?: string;
  template: FieldInputTemplate;
  value: unknown;
  onChange: (value: unknown) => void;
}

const invalidProps = (invalid: boolean | undefined) => (invalid ? { 'aria-invalid': true } : {});

const toFiniteNumber = (raw: string): number | null => {
  if (raw.trim() === '') {
    return null;
  }

  const parsed = Number(raw);

  return Number.isFinite(parsed) ? parsed : null;
};

const StringInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const text = typeof value === 'string' ? value : '';
  const onTextareaChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.currentTarget.value),
    [onChange]
  );
  const onInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => onChange(event.currentTarget.value),
    [onChange]
  );

  if (template.uiComponent === 'textarea') {
    return (
      <ResizableTextarea
        aria-label={template.title}
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
        onChange={onTextareaChange}
      />
    );
  }

  return (
    <Input
      aria-label={template.title}
      className="nodrag"
      id={id ? `${id}-input` : undefined}
      size="xs"
      value={text}
      w="full"
      {...invalidProps(invalid)}
      onChange={onInputChange}
    />
  );
};

const NumericInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const isInteger = template.type.name === 'IntegerField';
  const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : '';
  const min = template.minimum ?? template.exclusiveMinimum ?? undefined;
  const max = template.maximum ?? template.exclusiveMaximum ?? undefined;
  const onInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const parsed = toFiniteNumber(event.currentTarget.value);

      if (parsed !== null) {
        onChange(isInteger ? Math.round(parsed) : parsed);
      }
    },
    [isInteger, onChange]
  );

  return (
    <Input
      aria-label={template.title}
      className="nodrag"
      id={id ? `${id}-number-input` : undefined}
      max={max !== undefined ? String(max) : undefined}
      min={min !== undefined ? String(min) : undefined}
      size="xs"
      step={template.multipleOf !== null ? String(template.multipleOf) : isInteger ? '1' : 'any'}
      type="number"
      value={numericValue}
      w="full"
      {...invalidProps(invalid)}
      onChange={onInputChange}
    />
  );
};

const SWITCH_CHECKED_PROPS = { bg: 'accent.solid' };

const BooleanInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const onCheckedChange = useCallback((event: { checked: boolean }) => onChange(event.checked), [onChange]);

  return (
    <Switch.Root
      checked={value === true}
      className="nodrag"
      invalid={invalid}
      size="sm"
      onCheckedChange={onCheckedChange}
    >
      <Switch.HiddenInput
        aria-label={template.title}
        id={id ? `${id}-switch-input` : undefined}
        {...invalidProps(invalid)}
      />
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
  const selectedValue = useMemo(
    () => (typeof value === 'string' && options.some((option) => option.value === value) ? [value] : []),
    [options, value]
  );
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
        label: template.uiChoiceLabels?.[option] ?? option,
        value: option,
      })),
    [template.options, template.uiChoiceLabels]
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
        onValueChange={onChange}
      />
    );
  }

  return (
    <SelectInput id={id} invalid={invalid} options={options} title={template.title} value={value} onChange={onChange} />
  );
};

const DEFAULT_MODEL_TYPES: ModelTaxonomyType[] = ['main', 'vae', 'lora', 'controlnet', 't2i_adapter', 'ip_adapter'];

const ModelIdentifierInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const selectedKey =
    typeof (value as { key?: unknown } | null)?.key === 'string' ? (value as { key: string }).key : null;
  const modelTypes = (template.uiModelType ?? DEFAULT_MODEL_TYPES) as ModelTaxonomyType[];
  const allowedBases = template.uiModelBase;
  // ui_model_format narrows further within a base/type — e.g. a loader's main-model field
  // that accepts only diffusers-folder installs while its override fields take the
  // single-file checkpoints. Offering the wrong format here would enqueue a graph that
  // fails deep inside the model loader instead of at selection time.
  const allowedFormats = template.uiModelFormat;
  const filter = useCallback(
    (model: ModelConfig) =>
      (allowedBases ? allowedBases.includes(model.base) : true) &&
      (allowedFormats ? allowedFormats.includes(model.format) : true),
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
        filter={allowedBases || allowedFormats ? filter : undefined}
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
    fileAccept: 'video/*',
    getThumbnailUrl: (name: string) => galleryVideoUrls.thumbnail(name),
    nameKey: 'video_name',
    noun: 'video',
  },
} as const satisfies Record<
  WorkflowMediaKind,
  { fileAccept: string; getThumbnailUrl: (name: string) => string; nameKey: string; noun: string }
>;

const HIDDEN_FILE_INPUT_STYLE = { display: 'none' } as const;

/**
 * Direct input for `ImageField` / `VideoField`: shows the current item with a
 * thumbnail, adopts the gallery selection, accepts a single-item gallery drag
 * onto the row, and uploads a local file to the gallery's selected board.
 */
const MediaInput = ({ id, invalid, kind, onChange, value }: WorkflowFieldInputProps & { kind: WorkflowMediaKind }) => {
  const config = MEDIA_FIELD_CONFIG[kind];
  const selectedGalleryItem = useWorkflowProjectSelector((project) =>
    getWorkflowSelectedGalleryItem(project.galleryValues)
  );
  const gallerySelection = selectedGalleryItem?.kind === kind ? selectedGalleryItem : null;
  const uploadBoardId = useWorkflowProjectSelector((project) =>
    typeof project.galleryValues.selectedBoardId === 'string' ? project.galleryValues.selectedBoardId : 'none'
  );
  const mediaName =
    typeof (value as Record<string, unknown> | null | undefined)?.[config.nameKey] === 'string'
      ? ((value as Record<string, string>)[config.nameKey] ?? null)
      : null;
  const invalidAriaProps = useMemo(() => (invalid ? { 'aria-invalid': true } : {}), [invalid]);

  // dnd: the whole input row is a drop target for a single gallery item of the
  // matching kind. The instance-unique suffix keeps ids distinct when the node
  // editor and the Linear UI panel render the same field at once.
  const instanceId = useId();
  const dropId = getWorkflowMediaFieldDropId(`${id ?? 'field'}:${instanceId}`);
  const { active } = useDndContext();
  const acceptsActiveDrag = getWorkflowMediaFieldDropItem(active?.data.current, kind) !== null;
  const { isOver, setNodeRef } = useDroppable({ disabled: !acceptsActiveDrag, id: dropId });
  const onDragEnd = useCallback(
    (event: DragEndEvent) => {
      if (event.over?.id !== dropId) {
        return;
      }

      const item = getWorkflowMediaFieldDropItem(event.active.data.current, kind);

      if (item) {
        onChange({ [config.nameKey]: item.name });
      }
    },
    [config.nameKey, dropId, kind, onChange]
  );

  useDndMonitor({ onDragEnd });

  // Upload: file picker -> gallery upload -> adopt the uploaded item. The
  // adoption is pinned to this widget instance AND the project it started in:
  // `onChange` dispatches into the *active* project, so a completion arriving
  // after a project switch (or after this node was deleted, which unmounts the
  // widget) must not be applied - the upload itself still succeeded, so the
  // gallery is refreshed and the user is pointed there instead.
  const { project } = useWorkflowUi();
  const queryClient = useQueryClient();
  const isMountedRef = useRef(true);

  useEffect(
    () => () => {
      isMountedRef.current = false;
    },
    []
  );

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const onUploadClick = useCallback(() => fileInputRef.current?.click(), []);
  const onFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];

      // Reset so picking the same file again re-fires the change event.
      event.currentTarget.value = '';

      if (!file) {
        return;
      }

      // `accept` on the input is advisory only ("All Files" bypasses it); an
      // unknown type (empty string) is left for the server to judge.
      if (file.type && !file.type.startsWith(`${kind}/`)) {
        toaster.create({ title: `Please choose a ${config.noun} file`, type: 'error' });

        return;
      }

      const owner = captureAccountScope();
      const projectId = project.getSnapshot().id;

      setIsUploading(true);
      void (async () => {
        try {
          const name =
            kind === 'image'
              ? (await galleryTransfers.upload(file, uploadBoardId, { signal: owner.signal })).imageName
              : (await galleryTransfers.uploadVideo(file, uploadBoardId, { signal: owner.signal })).name;

          assertAccountScopeCurrent(owner);
          void invalidateGallery(queryClient, owner);

          if (isMountedRef.current && project.getSnapshot().id === projectId) {
            onChange({ [config.nameKey]: name });
          } else {
            toaster.create({
              description: `The workflow changed while ${name} uploaded - find it in the gallery.`,
              title: 'Upload finished',
              type: 'info',
            });
          }
        } catch {
          if (isAccountScopeCurrent(owner)) {
            toaster.create({ title: `Failed to upload ${config.noun}`, type: 'error' });
          }
        } finally {
          if (isAccountScopeCurrent(owner) && isMountedRef.current) {
            setIsUploading(false);
          }
        }
      })();
    },
    [config.nameKey, config.noun, kind, onChange, project, queryClient, uploadBoardId]
  );

  const onUseGallerySelectionClick = useCallback(() => {
    if (gallerySelection) {
      onChange({ [config.nameKey]: gallerySelection.name });
    }
  }, [config.nameKey, gallerySelection, onChange]);
  const onClearClick = useCallback(() => onChange(undefined), [onChange]);

  // A stale value (media deleted since the workflow was saved) 404s the
  // thumbnail; degrade to a media icon rather than the broken-image glyph. A
  // new value retries.
  const [failedThumbnail, setFailedThumbnail] = useState<string | null>(null);
  const onThumbnailError = useCallback(() => setFailedThumbnail(mediaName), [mediaName]);

  // Resolve the item's details for the dimensions/duration badge (the legacy
  // editor's widget shows the same). Best-effort: the preview works from the
  // name alone, so a failed lookup just drops the badge.
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

  return (
    <Box position="relative" w="full" {...invalidAriaProps}>
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
        {mediaName ? (
          <>
            <Flex
              alignItems="center"
              borderWidth="1px"
              h="full"
              justifyContent="center"
              overflow="hidden"
              rounded="sm"
              w="full"
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
            {badge !== null ? (
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
          </>
        ) : (
          <Flex
            alignItems="center"
            borderStyle="dashed"
            borderWidth="1px"
            h="full"
            justifyContent="center"
            rounded="sm"
            w="full"
          >
            <Text color="fg.subtle" fontSize="xs">
              {`Drop a ${config.noun} here`}
            </Text>
          </Flex>
        )}
        {acceptsActiveDrag ? (
          <DropZone
            alignItems="center"
            display="flex"
            inset="0"
            isOver={isOver}
            justifyContent="center"
            pointerEvents="none"
            position="absolute"
            variant="overlay"
          >
            <Text fontSize="xs" fontWeight="700">
              {`Drop ${config.noun}`}
            </Text>
          </DropZone>
        ) : null}
      </Box>
      <HStack gap="1.5" mt="1" w="full">
        <Button
          className="nodrag"
          disabled={!gallerySelection}
          size="2xs"
          title={gallerySelection ? `Use ${gallerySelection.name}` : `Select a ${config.noun} in the Gallery first.`}
          variant="outline"
          onClick={onUseGallerySelectionClick}
        >
          Use gallery selection
        </Button>
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
 * Integer input for `ui_component=video-frame-index` fields (`frame_index` on
 * Frame from Video; `start_frame`/`end_frame` on Frame Range from Video): the
 * standard number input plus a live frame preview and a scrubber slider, all
 * writing the same field value. The preview is a muted `<video>` element
 * seeked to `frame / fps` — browsers display the frame natively without a
 * canvas roundtrip.
 *
 * Degrades to the plain number input (plus a hint) when the companion video
 * field is unset/connection-driven or the video has no probed frame rate.
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

  // Frame count is the slider's upper bound. duration*fps can be off-by-one for
  // VFR containers, but the slider is for visual scrubbing — the backend
  // re-resolves indices against the authoritative decoder count at invoke time.
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

  // Seek whenever the resolved index changes. The half-frame nudge lands the
  // seek inside the frame's display window — some codecs decode the exact
  // boundary as black on first paint without it.
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

/**
 * Workflow `ColorField` values carry alpha as a `[0, 255]` integer, unlike
 * every other color in the app (and unlike `RgbaColor`, whose alpha is a unit
 * float). The scaling stays local to this adapter rather than pushing a second
 * alpha convention into `@platform/ui`'s color helpers.
 */
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
 * The `Apply LoRA Collection` loaders take `LoRAField | list[LoRAField]`. Rather than making the
 * user wire up one `Select LoRA` node per LoRA and collect them, this edits the list in place: pick
 * a LoRA from the model picker to append it, then tune each weight on its own row.
 *
 * Rows address entries by index, not by model key. The picker keeps the user from adding a
 * duplicate, but an imported workflow can already contain one, and a key-based edit would hit every
 * copy at once. Indexing also keeps entries this widget cannot read (`null` holes from
 * `normalizeLoraFieldCollectionValue`) in place instead of dropping them on the next edit.
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
  // An emptied list is written back as `undefined` rather than `[]`: it is the loaders' own default,
  // so the node returns to "no LoRAs" instead of sitting on a value that is equal in effect but
  // different from its default.
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
    (index: number, weight: number) =>
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
  onWeightChange: (index: number, weight: number) => void;
}) => {
  const label = entry ? entry.lora.name : 'Unreadable entry';
  // The committed value is a number, so re-rendering from it alone would rewrite a half-typed
  // "0." to "0" mid-keystroke and turn the next digit into "05" — a 10x wrong weight from a
  // plausible typing sequence. The draft holds the raw text until the field is left; blur drops
  // it so the row picks up the committed (and range-clamped) value again.
  const [draft, setDraft] = useState<string | null>(null);
  const onRemoveClick = useCallback(() => onRemove(index), [index, onRemove]);
  const onValueChange = useCallback(
    ({ value: valueAsText, valueAsNumber }: NumberInput.ValueChangeDetails) => {
      setDraft(valueAsText);

      if (Number.isFinite(valueAsNumber)) {
        onWeightChange(index, valueAsNumber);
      }
    },
    [index, onWeightChange]
  );
  const onFocusChange = useCallback(({ focused }: NumberInput.FocusChangeDetails) => {
    if (!focused) {
      setDraft(null);
    }
  }, []);

  return (
    <HStack gap="1" minW="0" w="full">
      <Tooltip content={entry ? label : 'This entry is not a readable LoRA and will block invoking.'}>
        <MiddleTruncate color={entry ? undefined : 'fg.error'} flex="1" fontSize="2xs" minW="0" text={label} />
      </Tooltip>
      {entry ? (
        <NumberInput.Root
          className="nodrag"
          // Stated rather than left to the default: `min`/`max` are otherwise advisory here, and a
          // mistyped 999 would run — the backend takes an unbounded float and nothing downstream
          // rejects it.
          clampValueOnBlur
          flexShrink="0"
          max={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
          min={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
          size="xs"
          step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
          value={draft ?? String(entry.weight)}
          w="16"
          onFocusChange={onFocusChange}
          onValueChange={onValueChange}
        >
          <NumberInput.Input
            aria-label={`${label} weight`}
            fontVariantNumeric="tabular-nums"
            id={id ? `${id}-lora-${index}-weight` : undefined}
          />
        </NumberInput.Root>
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

const CONNECTION_ONLY_FALLBACK = (
  <Text color="fg.subtle" fontSize="2xs">
    Connection only
  </Text>
);

export const WorkflowFieldInput = (props: WorkflowFieldInputProps) => {
  // COLLECTION media fields (e.g. Concatenate Videos' list input) hold arrays;
  // the single-value media widget would write a bare object into them. The
  // node editor never shows a control for COLLECTION fields, but linear-form
  // elements migrated from legacy `exposedFields` can reach here directly.
  if (
    (props.template.type.name === 'ImageField' || props.template.type.name === 'VideoField') &&
    props.template.type.cardinality === 'COLLECTION'
  ) {
    return CONNECTION_ONLY_FALLBACK;
  }

  switch (props.template.type.name) {
    case 'StringField':
      return <StringInput {...props} />;
    case 'IntegerField':
      if (props.template.uiComponent === 'video-frame-index' && props.template.type.cardinality === 'SINGLE') {
        return <VideoFrameIndexInput {...props} />;
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
