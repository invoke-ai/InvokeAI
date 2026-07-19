import type {
  NumberInput as ChakraNumberInput,
  SelectValueChangeDetails,
  SliderValueChangeDetails,
} from '@chakra-ui/react';
import type { ModelConfig, ModelTaxonomyType } from '@features/models';
import type { FilterParamSpec } from '@workbench/canvas-operations/api';
import type { ChangeEvent } from 'react';

import { Box, createListCollection, HStack, Input, NumberInput, Switch, Text } from '@chakra-ui/react';
import { ModelSelect } from '@features/models/react';
import { Field, Select, Slider } from '@platform/ui';
import {
  CONTROL_FILTERS,
  getFilterDefinition,
  getFilterNumberBounds,
  isSpandrelModelIdentifier,
} from '@workbench/canvas-operations/api';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const SELECT_POSITIONING_DOWN = { placement: 'bottom-end', sameWidth: false } as const;
const SELECT_POSITIONING_UP = { placement: 'top-end', sameWidth: false } as const;
const SPANDREL_MODEL_TYPES: ModelTaxonomyType[] = ['spandrel_image_to_image'];

interface LayerFilterControlsProps {
  filterType: string;
  settings: Record<string, unknown>;
  disabled: boolean;
  focusFilter: boolean;
  variant?: LayerFilterControlsVariant;
  onFilterTypeChange(value: string): void;
  onSettingsChange(value: Record<string, unknown>): void;
}

type LayerFilterControlsVariant = 'operation' | 'property';

export const getLayerFilterControlPolicy = (variant: LayerFilterControlsVariant) =>
  variant === 'operation'
    ? ({
        controlMinH: undefined,
        controlSize: 'xs',
        fieldOrientation: 'horizontal',
        fieldW: { enum: '13rem', filter: '11rem', model: '16rem', number: '17rem', string: '13rem' },
        modelSize: 'xs',
        positioning: SELECT_POSITIONING_UP,
        showFilterLabel: false,
        showNumberStepper: false,
      } as const)
    : ({
        controlMinH: undefined,
        controlSize: 'xs',
        fieldOrientation: 'vertical',
        fieldW: undefined,
        modelSize: 'xs',
        positioning: SELECT_POSITIONING_DOWN,
        showFilterLabel: true,
        showNumberStepper: true,
      } as const);

export const LayerFilterControls = ({
  disabled,
  filterType,
  focusFilter,
  onFilterTypeChange,
  onSettingsChange,
  settings,
  variant = 'property',
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
  const policy = getLayerFilterControlPolicy(variant);
  const filterFieldW = policy.fieldW?.filter;
  const filterTriggerProps = useMemo(
    () => ({ autoFocus: focusFilter, minH: policy.controlMinH }),
    [focusFilter, policy.controlMinH]
  );

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

  const filterSelect = (
    <Select
      aria-label={t('widgets.layers.control.filter')}
      collection={filterCollection}
      disabled={disabled}
      positioning={policy.positioning}
      size={policy.controlSize}
      triggerProps={filterTriggerProps}
      value={filterValue}
      valueText={t(`widgets.layers.control.filters.${filterType}`, filterType)}
      onValueChange={handleFilterTypeChange}
    />
  );

  return (
    <>
      {policy.showFilterLabel ? (
        <Field label={t('widgets.layers.control.filter')} w={filterFieldW}>
          {filterSelect}
        </Field>
      ) : (
        <Box flexShrink="0" w={filterFieldW}>
          {filterSelect}
        </Box>
      )}
      {definition?.params.map((param) => (
        <FilterParamField
          key={param.key}
          disabled={disabled}
          param={param}
          policy={policy}
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
  policy: ReturnType<typeof getLayerFilterControlPolicy>;
  settings: Record<string, unknown>;
  value: unknown;
  onChange: (key: string, value: unknown) => void;
}

const FilterParamField = ({ disabled, param, policy, settings, value, onChange }: FilterParamFieldProps) => {
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
  const enumTriggerProps = useMemo(() => ({ minH: policy.controlMinH }), [policy.controlMinH]);
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
        minH={policy.controlMinH}
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
      <Field label={label} orientation={policy.fieldOrientation} w={policy.fieldW?.enum}>
        <Select
          aria-label={label}
          collection={enumCollection}
          disabled={disabled}
          positioning={policy.positioning}
          size={policy.controlSize}
          triggerProps={enumTriggerProps}
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
      <Field label={label} orientation={policy.fieldOrientation} required w={policy.fieldW?.model}>
        <ModelSelect
          disabled={disabled}
          invalid={!model}
          modelTypes={SPANDREL_MODEL_TYPES}
          placeholder={label}
          size={policy.modelSize}
          value={model && typeof model.key === 'string' ? model.key : null}
          onChange={handleModel}
        />
      </Field>
    );
  }

  if (param.kind === 'string') {
    return (
      <Field label={label} orientation={policy.fieldOrientation} w={policy.fieldW?.string}>
        <Input
          disabled={disabled}
          minH={policy.controlMinH}
          size={policy.controlSize}
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
    <Field label={label} orientation={policy.fieldOrientation} w={policy.fieldW?.number}>
      <HStack gap="2">
        <Slider
          aria-label={labelAria}
          disabled={disabled}
          flex="1"
          max={numberBounds.sliderMax}
          minH={policy.controlMinH}
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
          minH={policy.controlMinH}
          size={policy.controlSize}
          step={numberBounds.step}
          value={String(numberCurrent)}
          w={policy.showNumberStepper ? '20' : '14'}
          onValueChange={handleNumberInput}
        >
          {policy.showNumberStepper ? <NumberInput.Control /> : null}
          <NumberInput.Input aria-label={label} />
        </NumberInput.Root>
      </HStack>
    </Field>
  );
};
