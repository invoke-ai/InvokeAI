import type { NumberInput as ChakraNumberInput, SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasLayerSourceContract, ShapeToolOptions } from '@workbench/canvas-engine/api';

import { createListCollection, HStack, NumberInput, Text } from '@chakra-ui/react';
import { ColorPicker, Select, ToggleDot } from '@platform/ui';
import { MAX_SHAPE_STROKE_WIDTH } from '@workbench/canvas-engine/api';
import { useShapeOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;
type ShapeKind = ShapeToolOptions['kind'];

interface SelectedShape {
  id: string;
  source: ShapeSource;
}

/** Fallback color used when re-enabling a `none` fill/stroke. */
const FALLBACK_COLOR = '#000000';

const SELECT_POSITIONING = { placement: 'top-start', sameWidth: false } as const;
const SELECT_TRIGGER_PROPS = { minW: '6rem' } as const;

/**
 * Shape tool options: kind (rect/ellipse), fill color (+ none), stroke color
 * (+ none) and stroke width. Edits set the defaults for the next created shape
 * AND, when a shape layer is selected, apply to it — colors commit ONE history
 * entry on interaction end (the ColorPicker popover shows the live color while
 * dragging; the canvas updates on release), discrete edits commit immediately.
 */
export const ShapeOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useShapeOptions(engine);

  const selected = useActiveProjectSelector(
    (project): SelectedShape | null => {
      const { document } = project.canvas;
      const layer = document.selectedLayerId
        ? document.layers.find((entry) => entry.id === document.selectedLayerId)
        : undefined;
      if (layer && layer.type === 'raster' && layer.source.type === 'shape') {
        return { id: layer.id, source: layer.source };
      }
      return null;
    },
    // Re-render on a selection change or a source-reference change (the reducer
    // replaces the source object on every edit).
    (a, b) => a?.id === b?.id && a?.source === b?.source
  );

  // Displayed values track the selected shape layer, or the tool defaults.
  const kind: ShapeKind = selected ? (selected.source.kind === 'ellipse' ? 'ellipse' : 'rect') : options.kind;
  const fill = selected ? selected.source.fill : options.fill;
  const stroke = selected ? selected.source.stroke : options.stroke;
  const strokeWidth = selected ? selected.source.strokeWidth : options.strokeWidth;

  const kindCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: ShapeKind }>({
        items: [
          { label: t('widgets.canvas.toolOptions.shapeRect'), value: 'rect' },
          { label: t('widgets.canvas.toolOptions.shapeEllipse'), value: 'ellipse' },
        ],
      }),
    [t]
  );
  const kindValue = useMemo(() => [kind], [kind]);

  /**
   * Applies an options patch: always updates the defaults store; when a shape
   * layer is selected and `commit` is set, records one history entry on it.
   */
  const applyEdit = useCallback(
    (patch: Partial<ShapeToolOptions>, commit: boolean) => {
      engine.interaction.set('shapeOptions', { fill, kind, stroke, strokeWidth, ...patch });
      if (selected && commit) {
        const before = selected.source;
        const after: ShapeSource = { ...before, ...patch };
        engine.layers.commitStructural(
          t('widgets.canvas.toolOptions.shapeEdit'),
          { id: selected.id, source: after, type: 'updateCanvasLayerSource' },
          { id: selected.id, source: before, type: 'updateCanvasLayerSource' }
        );
      }
    },
    [engine, fill, kind, stroke, strokeWidth, selected, t]
  );

  const onKindChange = useCallback(
    ({ value }: SelectValueChangeDetails<{ label: string; value: ShapeKind }>) => {
      const next = value[0] as ShapeKind | undefined;
      if (next && next !== kind) {
        applyEdit({ kind: next }, true);
      }
    },
    [applyEdit, kind]
  );

  // Fill: a toggle for none, plus a color picker (live store on drag, one commit on end).
  const onFillToggle = useCallback(
    (checked: boolean) => applyEdit({ fill: checked ? (fill ?? FALLBACK_COLOR) : null }, true),
    [applyEdit, fill]
  );
  const onFillChange = useCallback((hex: string) => applyEdit({ fill: hex }, false), [applyEdit]);
  const onFillChangeEnd = useCallback((hex: string) => applyEdit({ fill: hex }, true), [applyEdit]);

  const onStrokeToggle = useCallback(
    (checked: boolean) => applyEdit({ stroke: checked ? (stroke ?? FALLBACK_COLOR) : null }, true),
    [applyEdit, stroke]
  );
  const onStrokeChange = useCallback((hex: string) => applyEdit({ stroke: hex }, false), [applyEdit]);
  const onStrokeChangeEnd = useCallback((hex: string) => applyEdit({ stroke: hex }, true), [applyEdit]);

  const onStrokeWidthChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        applyEdit({ strokeWidth: Math.max(0, Math.round(valueAsNumber)) }, true);
      }
    },
    [applyEdit]
  );

  return (
    <HStack align="center" gap="3">
      <Select
        aria-label={t('widgets.canvas.toolOptions.shapeKind')}
        collection={kindCollection}
        positioning={SELECT_POSITIONING}
        size="xs"
        triggerProps={SELECT_TRIGGER_PROPS}
        value={kindValue}
        valueText={t(
          kind === 'ellipse' ? 'widgets.canvas.toolOptions.shapeEllipse' : 'widgets.canvas.toolOptions.shapeRect'
        )}
        onValueChange={onKindChange}
      />

      <HStack align="center" gap="1.5">
        <ToggleDot
          checked={fill !== null}
          label={t('widgets.canvas.toolOptions.shapeFill')}
          onCheckedChange={onFillToggle}
        />
        {fill !== null ? (
          <ColorPicker
            aria-label={t('widgets.canvas.toolOptions.shapeFill')}
            value={fill}
            onValueChange={onFillChange}
            onValueChangeEnd={onFillChangeEnd}
          />
        ) : null}
      </HStack>

      <HStack align="center" gap="1.5">
        <ToggleDot
          checked={stroke !== null}
          label={t('widgets.canvas.toolOptions.shapeStroke')}
          onCheckedChange={onStrokeToggle}
        />
        {stroke !== null ? (
          <>
            <ColorPicker
              aria-label={t('widgets.canvas.toolOptions.shapeStroke')}
              value={stroke}
              onValueChange={onStrokeChange}
              onValueChangeEnd={onStrokeChangeEnd}
            />
            <NumberInput.Root
              max={MAX_SHAPE_STROKE_WIDTH}
              min={0}
              size="xs"
              value={String(Math.round(strokeWidth))}
              w="4.5rem"
              onValueChange={onStrokeWidthChange}
            >
              <NumberInput.Control />
              <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.shapeStrokeWidth')} fontSize="xs" />
            </NumberInput.Root>
          </>
        ) : null}
      </HStack>

      <Text color="fg.muted" flexShrink="0" fontSize="2xs">
        {t('widgets.canvas.toolOptions.shapeHint')}
      </Text>
    </HStack>
  );
};
