import type { ModelConfig, ModelTaxonomyType } from '@features/models/react';
import type { FieldInputTemplate } from '@features/workflow/contracts';

import { Box, createListCollection, HStack, Image, Input, Switch, Text } from '@chakra-ui/react';
import { useDndContext, useDndMonitor, useDroppable, type DragEndEvent } from '@dnd-kit/core';
import { galleryDestinations, galleryTransfers, type GalleryBoard } from '@features/gallery';
import { getSelectedGalleryImageFromValues, getSelectedGalleryItemFromValues } from '@features/gallery/contracts';
import { invalidateGallery } from '@features/gallery/queries';
import { galleryImageUrls, galleryVideoUrls } from '@features/gallery/utility';
import { SCHEDULER_OPTIONS } from '@features/generation/settings';
import {
  getWorkflowMediaFieldDropId,
  getWorkflowMediaFieldDropItem,
  type WorkflowMediaKind,
} from '@features/workflow/ui/fields/mediaFieldDnd';
import { useWorkflowProjectSelector, useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
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
  parseHexColor,
  ResizableTextarea,
  Select,
  toaster,
} from '@platform/ui';
import { useQueryClient } from '@tanstack/react-query';
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
  const filter = useCallback(
    (model: ModelConfig) => (allowedBases ? allowedBases.includes(model.base) : true),
    [allowedBases]
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
        filter={allowedBases ? filter : undefined}
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
  // thumbnail; hide the broken-image glyph and keep showing the name. A new
  // value retries.
  const [failedThumbnail, setFailedThumbnail] = useState<string | null>(null);
  const onThumbnailError = useCallback(() => setFailedThumbnail(mediaName), [mediaName]);

  return (
    <Box ref={setNodeRef} position="relative" w="full">
      <HStack
        boxShadow={invalid ? '0 0 0 1px {colors.red.solid}' : undefined}
        gap="1.5"
        minW="0"
        rounded="sm"
        w="full"
        {...invalidAriaProps}
      >
        {mediaName ? (
          <>
            {failedThumbnail !== mediaName ? (
              <Image
                alt=""
                boxSize="6"
                flexShrink={0}
                objectFit="cover"
                rounded="xs"
                src={config.getThumbnailUrl(mediaName)}
                onError={onThumbnailError}
              />
            ) : null}
            <Text color="fg.muted" flex="1" fontSize="2xs" minW="0" title={mediaName} truncate>
              {mediaName}
            </Text>
          </>
        ) : (
          <Text color="fg.subtle" flex="1" fontSize="2xs">
            {`No ${config.noun} set`}
          </Text>
        )}
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
          <Text fontSize="2xs" fontWeight="700">
            {`Drop ${config.noun}`}
          </Text>
        </DropZone>
      ) : null}
    </Box>
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
    case 'FloatField':
      return <NumericInput {...props} />;
    case 'BooleanField':
      return <BooleanInput {...props} />;
    case 'EnumField':
      return <EnumInput {...props} />;
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
