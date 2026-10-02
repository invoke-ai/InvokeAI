import type { ReactNode } from 'react';

import { Badge, chakra, Flex, Grid, Icon, Stack, Switch, Text } from '@chakra-ui/react';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { ChevronDownIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { setPropertyGroupCollapsed, usePropertyGroupCollapsed } from './propertyGroupStore';

const GROUP_HEADER_HOVER = { color: 'fg' } as const;

/**
 * Use fixed label/trailing columns around a flexible control so sliders and numbers never wrap; blank labels
 * preserve continuation alignment.
 */
export const PropertyControlRow = ({ children, label }: { children: ReactNode; label?: string }) => (
  <Grid alignItems="center" columnGap="2" gridTemplateColumns="4.5rem minmax(0, 1fr) auto" minH="7" w="full">
    <Text color="fg.muted" fontSize="xs" minW="0" title={label} truncate>
      {label}
    </Text>
    {children}
  </Grid>
);

/** A labelled on/off row; the label is the switch's own, so the whole row is one click target. */
export const PropertySwitchRow = ({
  checked,
  disabled,
  label,
  onCheckedChange,
}: {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onCheckedChange: (checked: boolean) => void;
}) => {
  const handleChange = useCallback(
    ({ checked: next }: { checked: boolean }) => onCheckedChange(next),
    [onCheckedChange]
  );
  return (
    <Switch.Root
      checked={checked}
      disabled={disabled}
      justifyContent="space-between"
      minH="7"
      size="sm"
      w="full"
      onCheckedChange={handleChange}
    >
      <Switch.Label color="fg.muted" fontSize="xs">
        {label}
      </Switch.Label>
      <Switch.HiddenInput />
      <Switch.Control>
        <Switch.Thumb />
      </Switch.Control>
    </Switch.Root>
  );
};

/**
 * A named group of form rows. A collapsible group renders its label as a
 * disclosure button and remembers the user's choice per group id (an override
 * store; the declared default applies until the user touches it).
 */
export const PropertyGroup = ({
  children,
  collapsible,
  id,
  label,
}: {
  children: ReactNode;
  /** Absent means always open with a plain header. */
  collapsible?: 'open' | 'collapsed';
  id: string;
  label: string;
}) => {
  const collapsed = usePropertyGroupCollapsed(id, collapsible === 'collapsed');
  const onToggle = useCallback(() => setPropertyGroupCollapsed(id, !collapsed), [collapsed, id]);
  const open = !collapsible || !collapsed;
  return (
    <Stack aria-label={label} gap="1" role="group">
      {collapsible ? (
        <chakra.button
          alignItems="center"
          aria-expanded={open}
          color="fg.muted"
          display="flex"
          gap="1"
          rounded="xs"
          type="button"
          w="fit-content"
          _hover={GROUP_HEADER_HOVER}
          onClick={onToggle}
        >
          <Icon
            as={ChevronDownIcon}
            boxSize="3"
            transform={open ? undefined : 'rotate(-90deg)'}
            transitionDuration="fast"
            transitionProperty="transform"
          />
          <Text fontSize="xs" fontWeight="600">
            {label}
          </Text>
        </chakra.button>
      ) : (
        <Text color="fg.muted" fontSize="xs" fontWeight="600">
          {label}
        </Text>
      )}
      {open ? children : null}
    </Stack>
  );
};

/** A labelled choice row rendered as a full-width segmented control (≤5 options). */
export const PropertySegmentedRow = <Value extends string>({
  disabled,
  label,
  onValueChange,
  options,
  value,
}: {
  disabled?: boolean;
  label: string;
  onValueChange: (value: Value) => void;
  options: readonly { value: Value; label: string }[];
  value: Value;
}) => {
  const handleChange = useCallback(
    (next: string) => {
      if (next !== value) {
        onValueChange(next as Value);
      }
    },
    [onValueChange, value]
  );
  return (
    <PropertyControlRow label={label}>
      <SegmentedControl
        ariaLabel={label}
        disabled={disabled}
        gridColumn="2 / -1"
        options={options}
        value={value}
        onChange={handleChange}
      />
    </PropertyControlRow>
  );
};

/** Identify whether the form edits creation defaults or selected content. */
export const EditTargetChip = ({ layerName }: { layerName: string | null }) => {
  const { t } = useTranslation();
  return (
    <Flex justify="flex-end">
      <Badge
        colorPalette={layerName === null ? 'gray' : 'blue'}
        maxW="full"
        size="sm"
        title={layerName === null ? undefined : layerName}
        variant="surface"
      >
        <Text minW="0" truncate>
          {layerName === null
            ? t('widgets.properties.target.defaults')
            : t('widgets.properties.target.editing', { name: layerName })}
        </Text>
      </Badge>
    </Flex>
  );
};

export const HintCard = ({ rows }: { rows: readonly { gesture: string; effect: string }[] }) => (
  <Stack gap="1">
    {rows.map((row) => (
      <Grid key={row.gesture} alignItems="baseline" columnGap="2" gridTemplateColumns="auto minmax(0, 1fr)">
        <Badge colorPalette="gray" fontFamily="mono" size="sm" variant="surface">
          {row.gesture}
        </Badge>
        <Text color="fg.muted" fontSize="xs">
          {row.effect}
        </Text>
      </Grid>
    ))}
  </Stack>
);
