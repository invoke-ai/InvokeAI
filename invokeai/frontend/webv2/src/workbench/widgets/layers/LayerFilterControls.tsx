import type {
  NumberInput as ChakraNumberInput,
  SelectValueChangeDetails,
  SliderValueChangeDetails,
} from '@chakra-ui/react';
import type { FilterParamSpec } from '@workbench/generation/canvas/filterGraphs';
import type { ModelConfig, ModelTaxonomyType } from '@workbench/models/types';
import type { ChangeEvent } from 'react';

import { createListCollection, HStack, Input, NumberInput, Switch, Text } from '@chakra-ui/react';
import { Field, Select, Slider } from '@workbench/components/ui';
import {
  CONTROL_FILTERS,
  getFilterDefinition,
  getFilterNumberBounds,
  isSpandrelModelIdentifier,
} from '@workbench/generation/canvas/filterGraphs';
import { ModelSelect } from '@workbench/models/components/ModelSelect';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: false } as const;
const SPANDREL_MODEL_TYPES: ModelTaxonomyType[] = ['spandrel_image_to_image'];

interface LayerFilterControlsProps {
  filterType: string;
  settings: Record<string, unknown>;
  disabled: boolean;
  focusFilter: boolean;
  onFilterTypeChange(value: string): void;
  onSettingsChange(value: Record<string, unknown>): void;
}

export const LayerFilterControls = ({
  disabled,
  filterType,
  focusFilter,
  onFilterTypeChange,
  onSettingsChange,
  settings,
}: LayerFilterControlsProps) => {
  const { t } = useTranslation();
  const definition = getFilterDefinition(filterType);
  const filterCollection = useMemo(
    () =>
      createListCollection({
        items: CONTROL_FILTERS.map((filter) => ({
          label: t(`widgets.layers.control.filters.${filter.type}`, filter.type),
          value: filter.type,
        })),
      }),
    [t]
  );
  const filterValue = useMemo(() => [filterType], [filterType]);
  const filterTriggerProps = useMemo(() => ({ autoFocus: focusFilter }), [focusFilter]);

  const handleFilterTypeChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const next = value[0];
      if (next) {
        onFilterTypeChange(next);
      }
    },
    [onFilterTypeChange]
  );
  const handleSettingChange = useCallback(
    (key: string, value: unknown) => onSettingsChange({ ...settings, [key]: value }),
    [onSettingsChange, settings]
  );

  return (
    <>
      <Field label={t('widgets.layers.control.filter')}>
        <Select
          aria-label={t('widgets.layers.control.filter')}
          collection={filterCollection}
          disabled={disabled}
          positioning={SELECT_POSITIONING}
          size="xs"
          triggerProps={filterTriggerProps}
          value={filterValue}
          valueText={t(`widgets.layers.control.filters.${filterType}`, filterType)}
          onValueChange={handleFilterTypeChange}
        />
      </Field>
      {definition?.params.map((param) => (
        <FilterParamField
          key={param.key}
          disabled={disabled}
          param={param}
          settings={settings}
          value={settings[param.key]}
          onChange={handleSettingChange}
        />
      ))}
    </>
  );
};

interface FilterParamFieldProps {
  disabled: boolean;
  param: FilterParamSpec;
  settings: Record<string, unknown>;
  value: unknown;
  onChange: (key: string, value: unknown) => void;
}

const FilterParamField = ({ disabled, param, settings, value, onChange }: FilterParamFieldProps) => {
  const { t } = useTranslation();
  const label = t(`widgets.layers.control.filterParams.${param.key}`, param.key);
  const labelAria = useMemo(() => [label], [label]);

  const handleBoolean = useCallback(
    ({ checked }: { checked: boolean }) => onChange(param.key, checked),
    [onChange, param.key]
  );
  const handleEnum = useCallback(
    ({ value: next }: SelectValueChangeDetails) => {
      if (next[0]) {
        onChange(param.key, next[0]);
      }
    },
    [onChange, param.key]
  );
  const handleNumberEnd = useCallback(
    ({ value: next }: SliderValueChangeDetails) => {
      const n = next[0];
      if (n !== undefined && Number.isFinite(n)) {
        onChange(param.key, param.kind === 'number' && param.integer ? Math.round(n) : n);
      }
    },
    [onChange, param]
  );
  const handleNumberInput = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        onChange(param.key, param.kind === 'number' && param.integer ? Math.round(valueAsNumber) : valueAsNumber);
      }
    },
    [onChange, param]
  );
  const handleModel = useCallback(
    (model: ModelConfig | null) => {
      onChange(
        param.key,
        model ? { base: model.base, hash: model.hash, key: model.key, name: model.name, type: model.type } : null
      );
    },
    [onChange, param.key]
  );
  const handleString = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => onChange(param.key, event.currentTarget.value),
    [onChange, param.key]
  );

  const enumCollection = useMemo(
    () =>
      param.kind === 'enum'
        ? createListCollection({
            items: param.options.map((option) => ({ label: t(option.labelKey, option.value), value: option.value })),
          })
        : null,
    [param, t]
  );
  const enumCurrent =
    param.kind === 'enum' && typeof value === 'string' && param.options.some((option) => option.value === value)
      ? value
      : param.default;
  const enumCurrentOption = param.kind === 'enum' ? param.options.find((option) => option.value === enumCurrent) : null;
  const enumCurrentLabel = enumCurrentOption
    ? t(enumCurrentOption.labelKey, enumCurrentOption.value)
    : String(enumCurrent);
  const enumValue = useMemo(() => [String(enumCurrent)], [enumCurrent]);
  const numberCurrent = typeof value === 'number' && Number.isFinite(value) ? value : param.default;
  const numberBounds = param.kind === 'number' ? getFilterNumberBounds(param, settings) : null;
  const sliderCurrent = numberBounds
    ? Math.min(numberBounds.sliderMax, Math.max(numberBounds.sliderMin, Number(numberCurrent)))
    : Number(numberCurrent);
  const numberValue = useMemo(() => [sliderCurrent], [sliderCurrent]);

  if (param.kind === 'boolean') {
    return (
      <Switch.Root
        checked={typeof value === 'boolean' ? value : param.default}
        colorPalette="accent"
        disabled={disabled}
        size="xs"
        onCheckedChange={handleBoolean}
      >
        <Switch.HiddenInput />
        <Switch.Control>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label>
          <Text fontSize="xs">{label}</Text>
        </Switch.Label>
      </Switch.Root>
    );
  }

  if (param.kind === 'enum' && enumCollection) {
    return (
      <Field label={label}>
        <Select
          aria-label={label}
          collection={enumCollection}
          disabled={disabled}
          positioning={SELECT_POSITIONING}
          size="xs"
          value={enumValue}
          valueText={enumCurrentLabel}
          onValueChange={handleEnum}
        />
      </Field>
    );
  }

  if (param.kind === 'model') {
    const model = isSpandrelModelIdentifier(value) ? (value as { key: string }) : null;
    return (
      <Field label={label} required>
        <ModelSelect
          disabled={disabled}
          invalid={!model}
          modelTypes={SPANDREL_MODEL_TYPES}
          placeholder={label}
          size="xs"
          value={model && typeof model.key === 'string' ? model.key : null}
          onChange={handleModel}
        />
      </Field>
    );
  }

  if (param.kind === 'string') {
    return (
      <Field label={label}>
        <Input
          disabled={disabled}
          size="xs"
          value={typeof value === 'string' ? value : param.default}
          onChange={handleString}
        />
      </Field>
    );
  }

  if (param.kind !== 'number' || !numberBounds) {
    return null;
  }

  return (
    <Field label={label}>
      <HStack gap="2">
        <Slider
          aria-label={labelAria}
          disabled={disabled}
          flex="1"
          max={numberBounds.sliderMax}
          min={numberBounds.sliderMin}
          size="sm"
          step={numberBounds.step}
          value={numberValue}
          withThumbTooltip
          onValueChangeEnd={handleNumberEnd}
        />
        <NumberInput.Root
          disabled={disabled}
          max={numberBounds.inputMax}
          min={numberBounds.inputMin}
          size="xs"
          step={numberBounds.step}
          value={String(numberCurrent)}
          w="20"
          onValueChange={handleNumberInput}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={label} />
        </NumberInput.Root>
      </HStack>
    </Field>
  );
};
