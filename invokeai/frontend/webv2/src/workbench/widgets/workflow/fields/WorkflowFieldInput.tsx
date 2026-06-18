import type { ModelConfig, ModelTaxonomyType } from '@workbench/models/types';
import type { GeneratedImageContract } from '@workbench/types';
import type { FieldInputTemplate } from '@workbench/workflows/types';

import { HStack, Input, NativeSelect, Switch, Text, Textarea } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { listGalleryBoards, type GalleryBoard } from '@workbench/gallery/api';
import { SCHEDULER_OPTIONS } from '@workbench/generation/baseGenerationPolicies';
import { ModelSelect } from '@workbench/models/components';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useEffect, useState, type ChangeEvent } from 'react';

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
        onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.currentTarget.value)}
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
      onChange={(event: ChangeEvent<HTMLInputElement>) => onChange(event.currentTarget.value)}
    />
  );
};

const NumberInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const isInteger = template.type.name === 'IntegerField';
  const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : '';
  const min = template.minimum ?? template.exclusiveMinimum ?? undefined;
  const max = template.maximum ?? template.exclusiveMaximum ?? undefined;

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
      onChange={(event: ChangeEvent<HTMLInputElement>) => {
        const parsed = toFiniteNumber(event.currentTarget.value);

        if (parsed !== null) {
          onChange(isInteger ? Math.round(parsed) : parsed);
        }
      }}
    />
  );
};

const BooleanInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => (
  <Switch.Root
    checked={value === true}
    className="nodrag"
    invalid={invalid}
    size="sm"
    onCheckedChange={(event) => onChange(event.checked)}
  >
    <Switch.HiddenInput
      aria-label={template.title}
      id={id ? `${id}-switch-input` : undefined}
      {...invalidProps(invalid)}
    />
    <Switch.Control _checked={{ bg: 'accent.solid' }}>
      <Switch.Thumb />
    </Switch.Control>
  </Switch.Root>
);

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
}) => (
  <NativeSelect.Root className="nodrag" invalid={invalid} size="xs" w="full">
    <NativeSelect.Field
      aria-label={title}
      id={id ? `${id}-select` : undefined}
      value={typeof value === 'string' ? value : ''}
      onChange={(event: ChangeEvent<HTMLSelectElement>) => onChange(event.currentTarget.value)}
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

const EnumInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => (
  <SelectInput
    id={id}
    invalid={invalid}
    options={(template.options ?? []).map((option) => ({
      label: template.uiChoiceLabels?.[option] ?? option,
      value: option,
    }))}
    title={template.title}
    value={value}
    onChange={onChange}
  />
);

const DEFAULT_MODEL_TYPES: ModelTaxonomyType[] = ['main', 'vae', 'lora', 'controlnet', 't2i_adapter', 'ip_adapter'];

const ModelIdentifierInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const selectedKey =
    typeof (value as { key?: unknown } | null)?.key === 'string' ? (value as { key: string }).key : null;
  const modelTypes = (template.uiModelType as ModelTaxonomyType[] | null) ?? DEFAULT_MODEL_TYPES;
  const allowedBases = template.uiModelBase;

  return (
    <ModelSelect
      className="nodrag nowheel"
      filter={allowedBases ? (model: ModelConfig) => allowedBases.includes(model.base) : undefined}
      id={id ? `${id}-model-combobox` : undefined}
      invalid={invalid}
      modelTypes={modelTypes}
      size="xs"
      value={selectedKey}
      onChange={(model) =>
        onChange(
          model ? { base: model.base, hash: model.hash, key: model.key, name: model.name, type: model.type } : undefined
        )
      }
    />
  );
};

const SchedulerInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => (
  <SelectInput
    id={id}
    invalid={invalid}
    options={SCHEDULER_OPTIONS}
    title={template.title}
    value={value}
    onChange={onChange}
  />
);

let boardOptionsRequest: Promise<GalleryBoard[]> | null = null;

const getBoardOptions = (): Promise<GalleryBoard[]> => {
  if (!boardOptionsRequest) {
    boardOptionsRequest = listGalleryBoards()
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

  return (
    <SelectInput
      id={id}
      invalid={invalid}
      options={[
        { label: 'Auto', value: 'auto' },
        { label: 'None', value: 'none' },
        ...boards.map((board) => ({ label: board.name, value: board.id })),
      ]}
      title={template.title}
      value={selected}
      onChange={(next) => onChange(next === 'auto' || next === 'none' ? next : { board_id: next })}
    />
  );
};

const ImageInput = ({ invalid, onChange, value }: WorkflowFieldInputProps) => {
  const gallerySelection = useActiveProjectSelector(
    (project) => getProjectWidgetValues(project, 'gallery').selectedImage as GeneratedImageContract | null | undefined
  );
  const imageName =
    typeof (value as { image_name?: unknown } | null)?.image_name === 'string'
      ? (value as { image_name: string }).image_name
      : null;
  return (
    <HStack
      boxShadow={invalid ? '0 0 0 1px {colors.red.solid}' : undefined}
      gap="1.5"
      minW="0"
      rounded="sm"
      w="full"
      {...(invalid ? { 'aria-invalid': true } : {})}
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
        onClick={() => {
          if (gallerySelection) {
            onChange({ image_name: gallerySelection.imageName });
          }
        }}
      >
        Use gallery selection
      </Button>
      {imageName ? (
        <Button className="nodrag" size="2xs" variant="ghost" onClick={() => onChange(undefined)}>
          Clear
        </Button>
      ) : null}
    </HStack>
  );
};

const COLOR_CHANNELS = ['r', 'g', 'b', 'a'] as const;

const ColorInput = ({ id, invalid, onChange, value }: WorkflowFieldInputProps) => {
  const color = (typeof value === 'object' && value !== null ? value : {}) as Partial<Record<string, number>>;

  return (
    <HStack gap="1" w="full">
      {COLOR_CHANNELS.map((channel) => (
        <Input
          key={channel}
          aria-label={`Color ${channel.toUpperCase()}`}
          className="nodrag"
          id={id ? `${id}-color-${channel}-input` : undefined}
          max="255"
          min="0"
          placeholder={channel.toUpperCase()}
          size="xs"
          type="number"
          value={typeof color[channel] === 'number' ? color[channel] : ''}
          w="full"
          {...invalidProps(invalid)}
          onChange={(event: ChangeEvent<HTMLInputElement>) => {
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
          }}
        />
      ))}
    </HStack>
  );
};

export const WorkflowFieldInput = (props: WorkflowFieldInputProps) => {
  switch (props.template.type.name) {
    case 'StringField':
      return <StringInput {...props} />;
    case 'IntegerField':
    case 'FloatField':
      return <NumberInput {...props} />;
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
