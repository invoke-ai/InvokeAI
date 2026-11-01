import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { ControlType } from '@workbench/canvas-engine/api';

import { createListCollection, HStack, Text } from '@chakra-ui/react';
import { Select } from '@platform/ui/Select';
import { CONTROL_TYPES } from '@workbench/canvas-engine/api';
import { PropertyControlRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { formatPanelText, type ControlPanelModel, type PanelTone } from './controlPanelModel';
import { OptionNameSeparator } from './OptionNameSeparator';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: true } as const;
const TONE_COLORS: Record<PanelTone, string> = {
  default: 'fg',
  error: 'fg.error',
  muted: 'fg.muted',
  warning: 'fg.warning',
};
const PLACEHOLDER_PROPS = { color: 'fg.muted' } as const;
const INVALID_PLACEHOLDER_PROPS = { color: 'fg.warning' } as const;
// The recipe paints `aria-invalid` triggers as errors; a missing type only warns, so its rule is outranked here.
// The theme has no `border.warning`; warning borders use the warning foreground, as the warning input shell does.
const WARNING_TRIGGER_CSS = {
  '&[aria-invalid=true]': { borderColor: 'fg.warning' },
  '&[aria-invalid=true]:hover': { bg: 'bg.warning', borderColor: 'fg.warning' },
  '&[aria-invalid=true]:focus-visible': {
    '--focus-ring-color': 'var(--chakra-colors-fg-warning) !important',
    borderColor: 'fg.warning',
  },
} as const;

interface TypeItem {
  hint: string;
  hintColor: string;
  label: string;
  value: ControlType;
}

/** The Type row: what the image controls, each choice hinted with what is installed for it. */
export const ControlTypeRow = ({
  describedBy,
  disabled,
  model,
  onChange,
}: {
  /** What explains a missing type, read while the row warns. */
  describedBy: string;
  disabled: boolean;
  model: ControlPanelModel['type'];
  onChange(type: ControlType, label: string): void;
}) => {
  const { i18n, t } = useTranslation();
  const label = t('widgets.layers.control.rows.type');
  const collection = useMemo(
    () =>
      createListCollection<TypeItem>({
        items: CONTROL_TYPES.map((type) => ({
          hint: formatPanelText(t, model.hints[type].text, i18n.language),
          hintColor: TONE_COLORS[model.hints[type].tone],
          label: t(`widgets.layers.control.types.${type}`),
          value: type,
        })),
      }),
    [i18n.language, model.hints, t]
  );
  const value = useMemo(() => (model.value ? [model.value] : []), [model.value]);
  const triggerProps = useMemo(
    () =>
      model.invalid ? { 'aria-describedby': describedBy, 'aria-invalid': true, css: WARNING_TRIGGER_CSS } : undefined,
    [describedBy, model.invalid]
  );
  const handleChange = useCallback(
    ({ value: next }: SelectValueChangeDetails) => {
      const type = next[0] as ControlType | undefined;
      if (type && type !== model.value) {
        onChange(type, t(`widgets.layers.control.types.${type}`));
      }
    },
    [model.value, onChange, t]
  );
  return (
    <PropertyControlRow label={label}>
      <Select
        aria-label={label}
        collection={collection}
        disabled={disabled}
        gridColumn="2 / -1"
        minW="0"
        positioning={SELECT_POSITIONING}
        renderItem={renderTypeItem}
        triggerProps={triggerProps}
        value={value}
        valueText={
          model.value ? t(`widgets.layers.control.types.${model.value}`) : t('widgets.layers.control.typePlaceholder')
        }
        valueTextProps={model.value ? undefined : model.invalid ? INVALID_PLACEHOLDER_PROPS : PLACEHOLDER_PROPS}
        onValueChange={handleChange}
      />
    </PropertyControlRow>
  );
};

const renderTypeItem = (item: TypeItem) => (
  <HStack gap="2" justify="space-between" w="full">
    <Text truncate>
      {item.label}
      {/* Keeps the option's accessible name readable: "Depth, 2 models". */}
      <OptionNameSeparator />
    </Text>
    <Text color={item.hintColor} flexShrink={0} fontSize="sm">
      {item.hint}
    </Text>
  </HStack>
);
