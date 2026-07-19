import type { FieldInputTemplate } from '@features/workflow/contracts';
import type { WorkflowModel as ModelConfig } from '@features/workflow/ui/contracts';

import { HStack, Input, NativeSelect, Switch, Text, Textarea } from '@chakra-ui/react';
import { galleryDestinations, type GalleryBoard, type GeneratedImageContract } from '@features/gallery';
import { SCHEDULER_OPTIONS } from '@features/generation/settings';
import {
  WorkflowModelSelect as ModelSelect,
  useWorkflowProjectSelector,
} from '@features/workflow/ui/WorkflowUiContext';
import { Button, Combobox } from '@platform/ui';
import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';

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
      <Textarea
        aria-label={template.title}
        className="nodrag nowheel"
        fontFamily="mono"
        id={id ? `${id}-textarea` : undefined}
        minH="4.5rem"
        resize="vertical"
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
  const selectedValue = typeof value === 'string' ? value : '';
  const onSelectChange = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => onChange(event.currentTarget.value),
    [onChange]
  );

  return (
    <NativeSelect.Root className="nodrag" invalid={invalid} size="xs" w="full">
      <NativeSelect.Field
        aria-label={title}
        id={id ? `${id}-select` : undefined}
        value={selectedValue}
        onChange={onSelectChange}
      >
        {typeof value !== 'string' || !options.some((option) => option.value === value) ? (
          <option value="">Select…</option>
        ) : null}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </NativeSelect.Field>
      <NativeSelect.Indicator />
    </NativeSelect.Root>
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

const DEFAULT_MODEL_TYPES = ['main', 'vae', 'lora', 'controlnet', 't2i_adapter', 'ip_adapter'];

const ModelIdentifierInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const selectedKey =
    typeof (value as { key?: unknown } | null)?.key === 'string' ? (value as { key: string }).key : null;
  const modelTypes = template.uiModelType ?? DEFAULT_MODEL_TYPES;
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

let boardOptionsRequest: Promise<GalleryBoard[]> | null = null;

const getBoardOptions = (): Promise<GalleryBoard[]> => {
  if (!boardOptionsRequest) {
    boardOptionsRequest = galleryDestinations
      .list()
      .then((loadedBoards) => loadedBoards.filter((board) => board.kind === 'board'))
      .catch((error: unknown) => {
        boardOptionsRequest = null;
        throw error;
      });
  }

  return boardOptionsRequest;
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

const ImageInput = ({ invalid, onChange, value }: WorkflowFieldInputProps) => {
  const gallerySelection = useWorkflowProjectSelector(
    (project) => project.galleryValues.selectedImage as GeneratedImageContract | null | undefined
  );
  const imageName =
    typeof (value as { image_name?: unknown } | null)?.image_name === 'string'
      ? (value as { image_name: string }).image_name
      : null;
  const invalidAriaProps = useMemo(() => (invalid ? { 'aria-invalid': true } : {}), [invalid]);
  const onUseGallerySelectionClick = useCallback(() => {
    if (gallerySelection) {
      onChange({ image_name: gallerySelection.imageName });
    }
  }, [gallerySelection, onChange]);
  const onClearClick = useCallback(() => onChange(undefined), [onChange]);

  return (
    <HStack
      boxShadow={invalid ? '0 0 0 1px {colors.red.solid}' : undefined}
      gap="1.5"
      minW="0"
      rounded="sm"
      w="full"
      {...invalidAriaProps}
    >
      {imageName ? (
        <Text color="fg.muted" flex="1" fontSize="2xs" minW="0" title={imageName} truncate>
          {imageName}
        </Text>
      ) : (
        <Text color="fg.subtle" flex="1" fontSize="2xs">
          No image set
        </Text>
      )}
      <Button
        className="nodrag"
        disabled={!gallerySelection}
        size="2xs"
        title={gallerySelection ? `Use ${gallerySelection.imageName}` : 'Select an image in the Gallery first.'}
        variant="outline"
        onClick={onUseGallerySelectionClick}
      >
        Use gallery selection
      </Button>
      {imageName ? (
        <Button className="nodrag" size="2xs" variant="ghost" onClick={onClearClick}>
          Clear
        </Button>
      ) : null}
    </HStack>
  );
};

const COLOR_CHANNELS = ['r', 'g', 'b', 'a'] as const;

const ColorInput = ({ id, invalid, onChange, value }: WorkflowFieldInputProps) => {
  const color = useMemo(
    () => (typeof value === 'object' && value !== null ? value : {}) as Partial<Record<string, number>>,
    [value]
  );

  return (
    <HStack gap="1" w="full">
      {COLOR_CHANNELS.map((channel) => (
        <ColorChannelInput
          key={channel}
          channel={channel}
          color={color}
          id={id}
          invalid={invalid}
          onChange={onChange}
        />
      ))}
    </HStack>
  );
};

const ColorChannelInput = ({
  channel,
  color,
  id,
  invalid,
  onChange,
}: {
  channel: (typeof COLOR_CHANNELS)[number];
  color: Partial<Record<string, number>>;
  id?: string;
  invalid?: boolean;
  onChange: (value: unknown) => void;
}) => {
  const ariaLabel = useMemo(() => `Color ${channel.toUpperCase()}`, [channel]);
  const placeholder = channel.toUpperCase();
  const onInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const parsed = toFiniteNumber(event.currentTarget.value);

      if (parsed !== null) {
        onChange({
          a: 255,
          b: 0,
          g: 0,
          r: 0,
          ...color,
          [channel]: Math.min(255, Math.max(0, Math.round(parsed))),
        });
      }
    },
    [channel, color, onChange]
  );

  return (
    <Input
      aria-label={ariaLabel}
      className="nodrag"
      id={id ? `${id}-color-${channel}-input` : undefined}
      max="255"
      min="0"
      placeholder={placeholder}
      size="xs"
      type="number"
      value={typeof color[channel] === 'number' ? color[channel] : ''}
      w="full"
      {...invalidProps(invalid)}
      onChange={onInputChange}
    />
  );
};

export const WorkflowFieldInput = (props: WorkflowFieldInputProps) => {
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
      return <ImageInput {...props} />;
    case 'ColorField':
      return <ColorInput {...props} />;
    default:
      return (
        <Text color="fg.subtle" fontSize="2xs">
          Connection only
        </Text>
      );
  }
};
