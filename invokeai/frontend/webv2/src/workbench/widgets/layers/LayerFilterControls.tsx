import type { SelectValueChangeDetails, SliderValueChangeDetails } from '@chakra-ui/react';
import type { FilterParamSpec } from '@workbench/generation/canvas/filterGraphs';

import { createListCollection, Switch, Text } from '@chakra-ui/react';
import { Field, Select, Slider } from '@workbench/components/ui';
import { CONTROL_FILTERS, getFilterDefinition } from '@workbench/generation/canvas/filterGraphs';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: false } as const;

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
  value: unknown;
  onChange: (key: string, value: unknown) => void;
}

const FilterParamField = ({ disabled, param, value, onChange }: FilterParamFieldProps) => {
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

  const enumCollection = useMemo(
    () =>
      param.kind === 'enum'
        ? createListCollection({ items: param.options.map((option) => ({ label: option, value: option })) })
        : null,
    [param]
  );
  const enumCurrent =
    param.kind === 'enum' && typeof value === 'string' && param.options.includes(value) ? value : param.default;
  const enumValue = useMemo(() => [String(enumCurrent)], [enumCurrent]);
  const numberCurrent = typeof value === 'number' && Number.isFinite(value) ? value : param.default;
  const numberValue = useMemo(() => [Number(numberCurrent)], [numberCurrent]);

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
          valueText={String(enumCurrent)}
          onValueChange={handleEnum}
        />
      </Field>
    );
  }

  return (
    <Field label={label}>
      <Slider
        aria-label={labelAria}
        disabled={disabled}
        max={param.kind === 'number' ? (param.max ?? 255) : 255}
        min={param.kind === 'number' ? (param.min ?? 0) : 0}
        size="sm"
        step={param.kind === 'number' ? (param.step ?? (param.integer ? 1 : 0.01)) : 0.01}
        value={numberValue}
        withThumbTooltip
        onValueChangeEnd={handleNumberEnd}
      />
    </Field>
  );
};
