import type { NumberInput as ChakraNumberInput, SelectValueChangeDetails } from '@chakra-ui/react';
import type { GradientStop, GradientToolOptions } from '@workbench/canvas-engine/engineStores';
import type { CanvasLayerSourceContract } from '@workbench/types';

import { createListCollection, HStack, NumberInput, Text } from '@chakra-ui/react';
import { ColorPicker, Select } from '@workbench/components/ui';
import { useGradientOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

type GradientSource = Extract<CanvasLayerSourceContract, { type: 'gradient' }>;
type GradientKind = GradientToolOptions['kind'];

interface SelectedGradient {
  id: string;
  source: GradientSource;
}

const SELECT_POSITIONING = { placement: 'top-start', sameWidth: false } as const;
const SELECT_TRIGGER_PROPS = { minW: '6rem' } as const;

const clamp01 = (v: number): number => Math.min(1, Math.max(0, v));

/** Splits a stop color into a `#rrggbb` part (for the picker) and an alpha in [0,1]. */
const splitColor = (color: string): { rgb: string; alpha: number } => {
  const match = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(color.trim());
  if (match) {
    return { alpha: match[2] !== undefined ? parseInt(match[2], 16) / 255 : 1, rgb: `#${match[1]}` };
  }
  return { alpha: 1, rgb: '#000000' };
};

/** Recombines a `#rrggbb` and an alpha into an `#rrggbbaa` string. */
const joinColor = (rgb: string, alpha: number): string => {
  const a = Math.round(clamp01(alpha) * 255)
    .toString(16)
    .padStart(2, '0');
  return `${rgb}${a}`;
};

/**
 * Gradient tool options: kind (linear/radial), angle (degrees), and a MINIMAL
 * two-stop editor — start/end color with per-stop opacity. Edits set defaults
 * for the next created gradient AND apply to a selected gradient layer (colors
 * commit one history entry on interaction end; discrete edits commit at once).
 * A full multi-stop editor is a follow-up.
 */
export const GradientOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useGradientOptions(engine);

  const selected = useActiveProjectSelector(
    (project): SelectedGradient | null => {
      const { document } = project.canvas;
      const layer = document.selectedLayerId
        ? document.layers.find((entry) => entry.id === document.selectedLayerId)
        : undefined;
      if (layer && layer.type === 'raster' && layer.source.type === 'gradient') {
        return { id: layer.id, source: layer.source };
      }
      return null;
    },
    (a, b) => a?.id === b?.id && a?.source === b?.source
  );

  const kind: GradientKind = selected ? selected.source.kind : options.kind;
  const angle = selected ? selected.source.angle : options.angle;
  const stops = selected ? selected.source.stops : options.stops;
  const start = stops[0] ?? { color: '#000000ff', offset: 0 };
  const end = stops[stops.length - 1] ?? { color: '#ffffffff', offset: 1 };
  const startParts = splitColor(start.color);
  const endParts = splitColor(end.color);

  const kindCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: GradientKind }>({
        items: [
          { label: t('widgets.canvas.toolOptions.gradientLinear'), value: 'linear' },
          { label: t('widgets.canvas.toolOptions.gradientRadial'), value: 'radial' },
        ],
      }),
    [t]
  );
  const kindValue = useMemo(() => [kind], [kind]);

  const applyGradient = useCallback(
    (next: { kind: GradientKind; angle: number; stops: GradientStop[] }, commit: boolean) => {
      engine.stores.gradientOptions.set({ angle: next.angle, kind: next.kind, stops: next.stops });
      if (selected && commit) {
        const before = selected.source;
        const after: GradientSource = { ...before, angle: next.angle, kind: next.kind, stops: next.stops };
        engine.layers.commitStructural(
          t('widgets.canvas.toolOptions.gradientEdit'),
          { id: selected.id, source: after, type: 'updateCanvasLayerSource' },
          { id: selected.id, source: before, type: 'updateCanvasLayerSource' }
        );
      }
    },
    [engine, selected, t]
  );

  const setStopColor = useCallback(
    (index: number, color: string, commit: boolean) => {
      const nextStops = stops.map((stop, i) => (i === index ? { ...stop, color } : stop));
      applyGradient({ angle, kind, stops: nextStops }, commit);
    },
    [angle, applyGradient, kind, stops]
  );

  const lastIndex = stops.length - 1;

  const onKindChange = useCallback(
    ({ value }: SelectValueChangeDetails<{ label: string; value: GradientKind }>) => {
      const next = value[0] as GradientKind | undefined;
      if (next && next !== kind) {
        applyGradient({ angle, kind: next, stops: [...stops] }, true);
      }
    },
    [angle, applyGradient, kind, stops]
  );

  const onAngleChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        applyGradient({ angle: Math.round(valueAsNumber), kind, stops: [...stops] }, true);
      }
    },
    [applyGradient, kind, stops]
  );

  const onStartColorChange = useCallback(
    (hex: string) => setStopColor(0, joinColor(hex, startParts.alpha), false),
    [setStopColor, startParts.alpha]
  );
  const onStartColorEnd = useCallback(
    (hex: string) => setStopColor(0, joinColor(hex, startParts.alpha), true),
    [setStopColor, startParts.alpha]
  );
  const onStartOpacity = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setStopColor(0, joinColor(startParts.rgb, clamp01(valueAsNumber / 100)), true);
      }
    },
    [setStopColor, startParts.rgb]
  );

  const onEndColorChange = useCallback(
    (hex: string) => setStopColor(lastIndex, joinColor(hex, endParts.alpha), false),
    [endParts.alpha, lastIndex, setStopColor]
  );
  const onEndColorEnd = useCallback(
    (hex: string) => setStopColor(lastIndex, joinColor(hex, endParts.alpha), true),
    [endParts.alpha, lastIndex, setStopColor]
  );
  const onEndOpacity = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setStopColor(lastIndex, joinColor(endParts.rgb, clamp01(valueAsNumber / 100)), true);
      }
    },
    [endParts.rgb, lastIndex, setStopColor]
  );

  return (
    <HStack align="center" gap="3">
      <Select
        aria-label={t('widgets.canvas.toolOptions.gradientKind')}
        collection={kindCollection}
        positioning={SELECT_POSITIONING}
        size="xs"
        triggerProps={SELECT_TRIGGER_PROPS}
        value={kindValue}
        valueText={t(
          kind === 'radial' ? 'widgets.canvas.toolOptions.gradientRadial' : 'widgets.canvas.toolOptions.gradientLinear'
        )}
        onValueChange={onKindChange}
      />

      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.gradientAngle')}
        </Text>
        <NumberInput.Root
          disabled={kind === 'radial'}
          max={360}
          min={-360}
          size="xs"
          value={String(Math.round(angle))}
          w="4.5rem"
          onValueChange={onAngleChange}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.gradientAngle')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>

      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.gradientStart')}
        </Text>
        <ColorPicker
          aria-label={t('widgets.canvas.toolOptions.gradientStart')}
          value={startParts.rgb}
          onValueChange={onStartColorChange}
          onValueChangeEnd={onStartColorEnd}
        />
        <NumberInput.Root
          max={100}
          min={0}
          size="xs"
          value={String(Math.round(startParts.alpha * 100))}
          w="4rem"
          onValueChange={onStartOpacity}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.gradientStartOpacity')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>

      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.gradientEnd')}
        </Text>
        <ColorPicker
          aria-label={t('widgets.canvas.toolOptions.gradientEnd')}
          value={endParts.rgb}
          onValueChange={onEndColorChange}
          onValueChangeEnd={onEndColorEnd}
        />
        <NumberInput.Root
          max={100}
          min={0}
          size="xs"
          value={String(Math.round(endParts.alpha * 100))}
          w="4rem"
          onValueChange={onEndOpacity}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.gradientEndOpacity')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
    </HStack>
  );
};
