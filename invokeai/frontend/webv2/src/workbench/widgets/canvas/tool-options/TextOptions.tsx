import type { NumberInput as ChakraNumberInput, SelectValueChangeDetails } from '@chakra-ui/react';
import type { TextToolOptions } from '@workbench/canvas-engine/engineStores';
import type { CanvasLayerSourceContract } from '@workbench/types';

import { createListCollection, HStack, NumberInput } from '@chakra-ui/react';
import {
  MAX_TEXT_FONT_SIZE,
  MIN_TEXT_FONT_SIZE,
  TEXT_FONT_FAMILIES,
  TEXT_FONT_WEIGHTS,
} from '@workbench/canvas-engine/engineStores';
import { ColorPicker, IconButton, Select } from '@workbench/components/ui';
import { useTextEditSession, useTextOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { AlignCenterIcon, AlignLeftIcon, AlignRightIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

type TextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;
type TextAlign = TextToolOptions['align'];

interface SelectedText {
  id: string;
  source: TextSource;
}

const SELECT_POSITIONING = { placement: 'top-start', sameWidth: false } as const;
const FAMILY_TRIGGER_PROPS = { minW: '7rem' } as const;
const WEIGHT_TRIGGER_PROPS = { minW: '4.5rem' } as const;

const ALIGN_ICONS: Record<TextAlign, typeof AlignLeftIcon> = {
  center: AlignCenterIcon,
  left: AlignLeftIcon,
  right: AlignRightIcon,
};

const ALIGN_LABEL_KEYS: Record<TextAlign, string> = {
  center: 'widgets.canvas.toolOptions.textAlignCenter',
  left: 'widgets.canvas.toolOptions.textAlignLeft',
  right: 'widgets.canvas.toolOptions.textAlignRight',
};

const ALIGN_VALUES: readonly TextAlign[] = ['left', 'center', 'right'];

/** One alignment icon button with a stable per-value click handler. */
const AlignButton = ({
  active,
  onSelect,
  value,
}: {
  active: boolean;
  onSelect: (value: TextAlign) => void;
  value: TextAlign;
}) => {
  const { t } = useTranslation();
  const Icon = ALIGN_ICONS[value];
  const onClick = useCallback(() => onSelect(value), [onSelect, value]);
  return (
    <IconButton
      aria-label={t(ALIGN_LABEL_KEYS[value])}
      aria-pressed={active}
      size="xs"
      variant={active ? 'solid' : 'ghost'}
      onClick={onClick}
    >
      <Icon />
    </IconButton>
  );
};

/**
 * Text tool options: font family, size, weight, line-height, alignment and
 * color. The displayed values follow a precedence — an active text-editing
 * session's live source wins, else the selected text layer's source, else the
 * tool defaults.
 *
 * Edits always update the defaults store (seeding the next created layer) and
 * then, depending on context:
 * - **session open**: restyle the live session (`updateTextEditStyle`), folded
 *   into the session's single commit — the contenteditable restyles instantly.
 * - **text layer selected, no session**: commit ONE `updateCanvasLayerSource`
 *   per completed change (documentPatch history — same pattern as ShapeOptions).
 *   Colors update the store live while dragging and commit once on release.
 * - **neither**: only the defaults change.
 */
export const TextOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useTextOptions(engine);
  const session = useTextEditSession(engine);

  const selected = useActiveProjectSelector(
    (project): SelectedText | null => {
      const { document } = project.canvas;
      const layer = document.selectedLayerId
        ? document.layers.find((entry) => entry.id === document.selectedLayerId)
        : undefined;
      if (layer && layer.type === 'raster' && layer.source.type === 'text') {
        return { id: layer.id, source: layer.source };
      }
      return null;
    },
    (a, b) => a?.id === b?.id && a?.source === b?.source
  );

  // Precedence: live session source → selected layer source → tool defaults.
  const active: TextToolOptions = session ? session.source : (selected?.source ?? options);
  const { align, color, fontFamily, fontSize, fontWeight, lineHeight } = active;

  const familyCollection = useMemo(
    () => createListCollection<{ label: string; value: string }>({ items: [...TEXT_FONT_FAMILIES] }),
    []
  );
  const familyValue = useMemo(() => [fontFamily], [fontFamily]);
  const familyLabel = useMemo(
    () => TEXT_FONT_FAMILIES.find((entry) => entry.value === fontFamily)?.label ?? fontFamily,
    [fontFamily]
  );

  const weightCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: string }>({
        items: TEXT_FONT_WEIGHTS.map((weight) => ({ label: String(weight), value: String(weight) })),
      }),
    []
  );
  const weightValue = useMemo(() => [String(fontWeight)], [fontWeight]);

  /**
   * Applies a style patch: always updates the defaults store; restyles the live
   * session when one is open; else records one history entry on the selected
   * text layer when `commit` is set.
   */
  const applyEdit = useCallback(
    (patch: Partial<TextToolOptions>, commit: boolean) => {
      engine.stores.textOptions.set({ align, color, fontFamily, fontSize, fontWeight, lineHeight, ...patch });
      if (session) {
        engine.updateTextEditStyle(patch);
        return;
      }
      if (selected && commit) {
        const before = selected.source;
        const after: TextSource = { ...before, ...patch };
        engine.commitStructural(
          t('widgets.canvas.toolOptions.textEdit'),
          { id: selected.id, source: after, type: 'updateCanvasLayerSource' },
          { id: selected.id, source: before, type: 'updateCanvasLayerSource' }
        );
      }
    },
    [engine, align, color, fontFamily, fontSize, fontWeight, lineHeight, session, selected, t]
  );

  const onFamilyChange = useCallback(
    ({ value }: SelectValueChangeDetails<{ label: string; value: string }>) => {
      const next = value[0];
      if (next && next !== fontFamily) {
        applyEdit({ fontFamily: next }, true);
      }
    },
    [applyEdit, fontFamily]
  );

  const onWeightChange = useCallback(
    ({ value }: SelectValueChangeDetails<{ label: string; value: string }>) => {
      const next = value[0] ? Number(value[0]) : undefined;
      if (next && next !== fontWeight) {
        applyEdit({ fontWeight: next }, true);
      }
    },
    [applyEdit, fontWeight]
  );

  const onSizeChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        const clamped = Math.min(MAX_TEXT_FONT_SIZE, Math.max(MIN_TEXT_FONT_SIZE, Math.round(valueAsNumber)));
        applyEdit({ fontSize: clamped }, true);
      }
    },
    [applyEdit]
  );

  const onLineHeightChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        applyEdit({ lineHeight: Math.max(0.5, Math.round(valueAsNumber * 10) / 10) }, true);
      }
    },
    [applyEdit]
  );

  const onAlign = useCallback((next: TextAlign) => applyEdit({ align: next }, true), [applyEdit]);
  const onColorChange = useCallback((hex: string) => applyEdit({ color: hex }, false), [applyEdit]);
  const onColorChangeEnd = useCallback((hex: string) => applyEdit({ color: hex }, true), [applyEdit]);

  return (
    <HStack align="center" gap="3">
      <Select
        aria-label={t('widgets.canvas.toolOptions.textFont')}
        collection={familyCollection}
        positioning={SELECT_POSITIONING}
        size="xs"
        triggerProps={FAMILY_TRIGGER_PROPS}
        value={familyValue}
        valueText={familyLabel}
        onValueChange={onFamilyChange}
      />

      <NumberInput.Root
        max={MAX_TEXT_FONT_SIZE}
        min={MIN_TEXT_FONT_SIZE}
        size="xs"
        value={String(Math.round(fontSize))}
        w="4.5rem"
        onValueChange={onSizeChange}
      >
        <NumberInput.Control />
        <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.textSize')} fontSize="xs" />
      </NumberInput.Root>

      <Select
        aria-label={t('widgets.canvas.toolOptions.textWeight')}
        collection={weightCollection}
        positioning={SELECT_POSITIONING}
        size="xs"
        triggerProps={WEIGHT_TRIGGER_PROPS}
        value={weightValue}
        valueText={String(fontWeight)}
        onValueChange={onWeightChange}
      />

      <NumberInput.Root
        max={4}
        min={0.5}
        size="xs"
        step={0.1}
        value={lineHeight.toFixed(1)}
        w="4.5rem"
        onValueChange={onLineHeightChange}
      >
        <NumberInput.Control />
        <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.textLineHeight')} fontSize="xs" />
      </NumberInput.Root>

      <HStack align="center" gap="0.5">
        {ALIGN_VALUES.map((value) => (
          <AlignButton key={value} active={align === value} value={value} onSelect={onAlign} />
        ))}
      </HStack>

      <ColorPicker
        aria-label={t('widgets.canvas.toolOptions.textColor')}
        value={color}
        onValueChange={onColorChange}
        onValueChangeEnd={onColorChangeEnd}
      />
    </HStack>
  );
};
