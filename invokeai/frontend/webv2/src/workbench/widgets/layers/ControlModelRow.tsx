import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { ModelConfig } from '@features/models';
import type { ControlType } from '@workbench/canvas-engine/api';
import type { ControlModelResolution } from '@workbench/controlModelResolution';

import { Box, createListCollection, Stack, Text } from '@chakra-ui/react';
import { ensureStartersLoaded } from '@features/models';
import { Button } from '@platform/ui/Button';
import { Select } from '@platform/ui/Select';
import { Tooltip } from '@platform/ui/Tooltip';
import { deriveControlKind } from '@workbench/controlAdapters';
import { classifyControlModel } from '@workbench/controlTypes';
import { PropertyControlRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { formatPanelText, type ControlPanelModel, type PanelTone } from './controlPanelModel';
import { OptionNameSeparator } from './OptionNameSeparator';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: true } as const;
const TONE_COLORS: Record<PanelTone, string | undefined> = {
  default: undefined,
  error: 'fg.error',
  muted: 'fg.muted',
  warning: 'fg.warning',
};

type ModelGroup = 'purpose' | 'union' | 'unknown';

interface ModelItem {
  caption: string;
  disabled: boolean;
  group: ModelGroup;
  label: string;
  value: string;
}

const groupOf = (item: ModelItem): string => item.group;

/**
 * The models a typed layer can pick, grouped: made for its type (or its detector's family), every union model on the
 * base (one that does not cover the type is still a deliberate pick), then models whose purpose nothing names.
 * Models made for another type stay out; untyped layers pick nothing here.
 */
const groupModels = (resolution: ControlModelResolution): { group: ModelGroup; model: ModelConfig }[] => {
  const unions = resolution.usable.filter((model) => classifyControlModel(model).union);
  const purpose = resolution.candidates.filter((model) => !unions.includes(model));
  const unknown = resolution.usable.filter(
    (model) => !purpose.includes(model) && !unions.includes(model) && classifyControlModel(model).family === null
  );
  return [
    ...purpose.map((model) => ({ group: 'purpose' as const, model })),
    ...unions.map((model) => ({ group: 'union' as const, model })),
    ...unknown.map((model) => ({ group: 'unknown' as const, model })),
  ];
};

/** The Model row and its two-line caption: what was resolved, why, and the one action that fixes or undoes it. */
export const ControlModelRow = ({
  canManageModels,
  captionId,
  controlType,
  disabled,
  model,
  onFindModels,
  onPick,
  onReset,
  reidentifyPending,
  rememberedHint,
  resolution,
  starterSources,
}: {
  canManageModels: boolean;
  /** The caption's id, so the Model (and, when it warns, the Type) choice is described by it. */
  captionId: string;
  controlType: ControlType | null;
  disabled: boolean;
  model: ControlPanelModel['model'];
  onFindModels(): void;
  onPick(model: ModelConfig): void;
  onReset(): void;
  /** Listed disabled after the rest: they need re-identifying before a layer can use them. */
  reidentifyPending: readonly ModelConfig[];
  /** Said beside Reset when the pick came from memory, since it applies to other layers too. */
  rememberedHint: string | null;
  resolution: ControlModelResolution;
  /** Sources of the starter models; null until the catalog has loaded, when items leave their origin out. */
  starterSources: ReadonlySet<string> | null;
}) => {
  const { i18n, t } = useTranslation();
  const label = t('widgets.layers.control.model');
  const selectRef = useRef<HTMLDivElement | null>(null);
  const format = useCallback(
    (text: Parameters<typeof formatPanelText>[1]) => formatPanelText(t, text, i18n.language),
    [i18n.language, t]
  );

  const collection = useMemo(() => {
    const itemFor = (entry: ModelConfig, group: ModelGroup, disabled: boolean): ModelItem => {
      const kind = deriveControlKind(entry);
      const kindLabel = kind ? t(`widgets.layers.control.kinds.${kind}`) : entry.type;
      return {
        caption: disabled
          ? t('widgets.layers.control.needsReidentify')
          : starterSources
            ? t('widgets.layers.control.modelItemCaption', {
                kind: kindLabel,
                origin: t(
                  `widgets.layers.control.modelOrigin.${starterSources.has(entry.source) ? 'starter' : 'downloaded'}`
                ),
              })
            : kindLabel,
        disabled,
        group,
        label: entry.name,
        value: entry.key,
      };
    };
    return createListCollection<ModelItem>({
      items: [
        ...groupModels(resolution).map(({ group, model: entry }) => itemFor(entry, group, false)),
        ...reidentifyPending.map((entry) => itemFor(entry, 'unknown', true)),
      ],
    });
  }, [reidentifyPending, resolution, starterSources, t]);
  const renderGroupLabel = useCallback(
    (group: string) =>
      group === 'purpose'
        ? t('widgets.layers.control.modelGroups.purpose', {
            type: controlType ? t(`widgets.layers.control.typeNouns.${controlType}`) : '',
          })
        : t(`widgets.layers.control.modelGroups.${group}`),
    [controlType, t]
  );

  const value = useMemo(() => (resolution.model ? [resolution.model.key] : []), [resolution.model]);
  const handleChange = useCallback(
    ({ value: next }: SelectValueChangeDetails) => {
      const picked = resolution.usable.find((entry) => entry.key === next[0]);
      if (picked && picked.key !== resolution.model?.key) {
        onPick(picked);
      }
    },
    [onPick, resolution.model?.key, resolution.usable]
  );
  const handleAction = useCallback(() => {
    if (model.action === 'reset') {
      onReset();
      // The Reset link goes away with the pick; the choice it undid is where focus belongs.
      selectRef.current?.querySelector<HTMLElement>('[data-part="trigger"]')?.focus();
    } else {
      onFindModels();
    }
  }, [model.action, onFindModels, onReset]);

  // Whether a model came with Invoke is read from the starter catalog, fetched the first time the list opens.
  const handleOpenChange = useCallback(({ open }: { open: boolean }) => {
    if (open) {
      ensureStartersLoaded();
    }
  }, []);
  const triggerProps = useMemo(() => ({ 'aria-describedby': captionId }), [captionId]);
  const valueText = model.name ?? (model.valueText ? format(model.valueText) : '');
  const valueTextProps = useMemo(() => ({ color: TONE_COLORS[model.tone] }), [model.tone]);
  const showAction = model.action === 'reset' || (model.action !== null && canManageModels);
  const actionLabel =
    model.action === 'reset'
      ? t('widgets.layers.control.modelAction.reset')
      : model.action === 'find-one'
        ? t('widgets.layers.control.modelAction.findOne')
        : t('widgets.layers.control.modelAction.installStarters');
  const actionButton = showAction ? (
    <Button
      color="accent.fg"
      fontSize="sm"
      h="auto"
      minW="0"
      px="0"
      size="xs"
      textDecoration="underline"
      variant="plain"
      verticalAlign="baseline"
      onClick={handleAction}
    >
      {actionLabel}
    </Button>
  ) : null;

  return (
    <Stack gap="0">
      <PropertyControlRow label={label}>
        <Box ref={selectRef} gridColumn="2 / -1" minW="0">
          <Select
            aria-label={label}
            collection={collection}
            disabled={disabled}
            groupBy={groupOf}
            positioning={SELECT_POSITIONING}
            renderGroupLabel={renderGroupLabel}
            renderItem={renderModelItem}
            triggerProps={triggerProps}
            value={value}
            valueText={valueText}
            valueTextProps={valueTextProps}
            onOpenChange={handleOpenChange}
            onValueChange={handleChange}
          />
        </Box>
      </PropertyControlRow>
      <PropertyControlRow>
        <Box gridColumn="2 / -1" minW="0">
          {/* Always two lines tall, so the rows below never move as the explanation changes. */}
          <Text color={TONE_COLORS[model.captionTone]} fontSize="sm" h="2rem" lineClamp={2} lineHeight="1rem">
            <span id={captionId}>{model.caption ? format(model.caption) : null}</span>{' '}
            {actionButton && rememberedHint ? <Tooltip content={rememberedHint}>{actionButton}</Tooltip> : actionButton}
          </Text>
        </Box>
      </PropertyControlRow>
    </Stack>
  );
};

const renderModelItem = (item: ModelItem) => (
  <Stack gap="0" minW="0">
    <Text truncate>
      {item.label}
      {/* Keeps the option's accessible name readable: "name, caption". */}
      <OptionNameSeparator />
    </Text>
    <Text color="fg.muted" fontSize="sm" truncate>
      {item.caption}
    </Text>
  </Stack>
);
