import { createListCollection, Field, HStack, Input, Slider, Stack, Switch, Text } from '@chakra-ui/react';
import { Select } from '@platform/ui/Select';
import { useCallback, useId, useMemo, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { SettingFieldProps } from './contracts';

import { resolveSettingsText } from './contracts';
import { ModifiedSettingIndicator } from './ModifiedSettingIndicator';

export interface SettingControlProps extends Pick<SettingFieldProps, 'field' | 'surface'> {
  value: boolean | number | string;
  onChange: (value: boolean | number | string) => void;
  disabled?: boolean;
  isModified?: boolean;
}

const DIALOG_DIRECTION = { base: 'column', md: 'row' } as const;
const DIALOG_ALIGNMENT = { base: 'stretch', md: 'center' } as const;
const DIALOG_CONTROL_WIDTH = { base: 'full', md: '52' } as const;
const SELECT_POSITIONING = { strategy: 'fixed', sameWidth: true, hideWhenDetached: true } as const;

/** Presentation only. Normalization, persistence and side effects belong to the binding. */
export const SettingControl = ({ field, surface, value, onChange, disabled, isModified }: SettingControlProps) => {
  const { t } = useTranslation();
  const descriptionId = useId();
  const label = resolveSettingsText(field.label, t);
  const description = surface === 'dialog' && field.description ? resolveSettingsText(field.description, t) : undefined;
  const items = useMemo(
    () =>
      field.kind === 'select'
        ? field.options.map((option) => ({ ...option, label: resolveSettingsText(option.label, t) }))
        : [],
    [field, t]
  );
  const collection = useMemo(() => createListCollection({ items }), [items]);
  const selectValue = useMemo(() => [String(value)], [value]);
  const sliderValue = useMemo(() => [Number(value)], [value]);
  const sliderLabel = useMemo(() => [label], [label]);
  const handleCheckedChange = useCallback(({ checked }: { checked: boolean }) => onChange(checked), [onChange]);
  const handleSelectChange = useCallback(
    ({ value: next }: { value: string[] }) => {
      if (next[0] !== undefined) {
        onChange(next[0]);
      }
    },
    [onChange]
  );
  const handleSliderChange = useCallback(
    ({ value: next }: { value: number[] }) => {
      if (next[0] !== undefined) {
        onChange(next[0]);
      }
    },
    [onChange]
  );
  const handleNumberChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const next = event.currentTarget.valueAsNumber;
      if (Number.isFinite(next)) {
        onChange(next);
      }
    },
    [onChange]
  );

  if (field.kind === 'boolean') {
    return (
      <Switch.Root
        checked={value === true}
        disabled={disabled}
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        gap="4"
        w="full"
        size="sm"
        onCheckedChange={handleCheckedChange}
      >
        <Stack gap="1">
          <HStack gap="2">
            <Switch.Label fontSize={surface === 'quick' ? 'xs' : 'sm'} fontWeight="500">
              {label}
            </Switch.Label>
            {isModified ? <ModifiedSettingIndicator label={label} /> : null}
          </HStack>
          {description ? (
            <Text id={descriptionId} color="fg.muted" fontSize="xs">
              {description}
            </Text>
          ) : null}
        </Stack>
        <Switch.HiddenInput aria-describedby={description ? descriptionId : undefined} />
        <Switch.Control flexShrink={0}>
          <Switch.Thumb />
        </Switch.Control>
      </Switch.Root>
    );
  }

  return (
    <Field.Root
      disabled={disabled}
      display="flex"
      flexDirection={surface === 'quick' ? 'column' : DIALOG_DIRECTION}
      alignItems={surface === 'quick' ? 'stretch' : DIALOG_ALIGNMENT}
      justifyContent="space-between"
      gap="3"
    >
      <Stack gap="1" flex="1">
        <HStack gap="2">
          <Field.Label fontSize={surface === 'quick' ? 'xs' : 'sm'} fontWeight="500">
            {label}
          </Field.Label>
          {isModified ? <ModifiedSettingIndicator label={label} /> : null}
        </HStack>
        {description ? (
          <Field.HelperText color="fg.muted" fontSize="xs">
            {description}
          </Field.HelperText>
        ) : null}
      </Stack>
      {field.kind === 'select' ? (
        <Select
          collection={collection}
          value={selectValue}
          disabled={disabled}
          size={surface === 'quick' ? 'xs' : 'sm'}
          w={surface === 'quick' ? 'full' : DIALOG_CONTROL_WIDTH}
          flexShrink={0}
          portalled={false}
          // Fixed positioning escapes scrolling/clipped surfaces; staying inline preserves modal focus and ARIA.
          positioning={SELECT_POSITIONING}
          onValueChange={handleSelectChange}
        />
      ) : field.kind === 'slider' ? (
        <HStack w={surface === 'quick' ? 'full' : DIALOG_CONTROL_WIDTH} gap="3" flexShrink={0}>
          <Slider.Root
            aria-label={sliderLabel}
            disabled={disabled}
            value={sliderValue}
            min={field.min}
            max={field.max}
            step={field.step ?? 1}
            flex="1"
            size="sm"
            onValueChange={handleSliderChange}
          >
            <Slider.Control>
              <Slider.Track>
                <Slider.Range />
              </Slider.Track>
              <Slider.Thumbs />
            </Slider.Control>
          </Slider.Root>
          <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums" minW="6" textAlign="end">
            {value}
          </Text>
        </HStack>
      ) : field.kind === 'number' ? (
        <Input
          aria-label={label}
          type="number"
          min={field.min}
          max={field.max}
          step={field.step ?? 1}
          value={String(value)}
          disabled={disabled}
          size="sm"
          w="24"
          onChange={handleNumberChange}
        />
      ) : null}
    </Field.Root>
  );
};
