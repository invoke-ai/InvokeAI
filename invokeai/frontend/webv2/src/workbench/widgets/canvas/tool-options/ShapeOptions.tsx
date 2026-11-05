import type { CanvasLayerSourceContract, ShapeToolKind, ShapeToolOptions } from '@workbench/canvas-engine/api';
import type { ToolFormProps, ToolPropertyForm } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { Text } from '@chakra-ui/react';
import { ToggleIconButton } from '@platform/ui/Button';
import { ColorPicker } from '@platform/ui/ColorPicker';
import { ScrubberField } from '@platform/ui/ScrubberField';
import {
  DEFAULT_SHAPE_OPTIONS,
  MAX_SHAPE_STROKE_WIDTH,
  getDocumentLayer,
  shapeDrawsOnLayer,
} from '@workbench/canvas-engine/api';
import { SHAPE_KIND_ORDER, SHAPE_KIND_PRESENTATION } from '@workbench/widgets/canvas/canvasToolCommands';
import { useActiveColorCommands, useActiveColorPair } from '@workbench/widgets/canvas/color-system/useActiveColors';
import { useShapeOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useSliderGesture } from '@workbench/widgets/canvas/tool-presentation/FormControls';
import {
  EditTargetChip,
  PropertyControlRow,
  PropertySegmentedRow,
} from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { PaintBucketIcon, SquareIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;
type ShapeStylePatch = Partial<Pick<ShapeSource, 'fill' | 'stroke' | 'strokeWidth'>>;

interface SelectedShape {
  id: string;
  name: string;
  source: ShapeSource;
}

const FALLBACK_COLOR = '#000000';
const formatPx = (value: number): string => `${value}px`;

/** Where Place on = Selected layer sends the next shape, given the selection. */
type SelectedPlacement = 'none' | 'layer' | 'mask' | 'new-layer';

const setShapeOptions = (engine: ToolFormProps['engine'], patch: Partial<ShapeToolOptions>): void => {
  engine.interaction.set('shapeOptions', { ...engine.interaction.get('shapeOptions'), ...patch });
};

/** Settings for the next shape: what it is and where it lands. They never touch a selected layer. */
const ShapeToolSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const options = useShapeOptions(engine);
  const placement = useActiveProjectSelector((project): SelectedPlacement => {
    const { document } = project.canvas;
    const layer = document.selectedLayerId ? getDocumentLayer(document, document.selectedLayerId) : undefined;
    if (!layer) {
      return 'none';
    }
    if (layer.type === 'inpaint_mask' || layer.type === 'regional_guidance') {
      return 'mask';
    }
    return shapeDrawsOnLayer(layer) ? 'layer' : 'new-layer';
  });
  const kindOptions = useMemo(
    () =>
      SHAPE_KIND_ORDER.map((value) => ({
        icon: SHAPE_KIND_PRESENTATION[value].icon,
        label: t(SHAPE_KIND_PRESENTATION[value].labelKey),
        value,
      })),
    [t]
  );
  const onKindChange = useCallback((kind: ShapeToolKind) => setShapeOptions(engine, { kind }), [engine]);
  const targetOptions = useMemo(
    () => [
      { label: t('widgets.canvas.toolOptions.shapeTargetSelected'), value: 'selected' as const },
      { label: t('widgets.canvas.toolOptions.shapeTargetNew'), value: 'new' as const },
    ],
    [t]
  );
  const onTargetChange = useCallback(
    (target: ShapeToolOptions['target']) => setShapeOptions(engine, { target }),
    [engine]
  );
  const hintKey =
    options.target !== 'selected'
      ? null
      : placement === 'mask'
        ? 'widgets.canvas.toolOptions.shapeMaskHint'
        : placement === 'new-layer'
          ? 'widgets.canvas.toolOptions.shapeTargetFallbackHint'
          : null;
  return (
    <>
      <PropertySegmentedRow
        label={t('widgets.properties.rows.kind')}
        options={kindOptions}
        value={options.kind}
        onValueChange={onKindChange}
      />
      <PropertySegmentedRow
        label={t('widgets.canvas.toolOptions.shapeTarget')}
        options={targetOptions}
        value={options.target}
        onValueChange={onTargetChange}
      />
      {hintKey ? (
        <Text color="fg.muted" fontSize="xs">
          {t(hintKey)}
        </Text>
      ) : null}
    </>
  );
};

/**
 * Edit the selected shape layer's fill and stroke, or the tool defaults with pair colors when no shape layer is
 * selected. Each gesture on a selected shape commits one undoable step.
 */
const useShapeAppearance = (engine: ToolFormProps['engine']) => {
  const { t } = useTranslation();
  const commitPrepared = usePreparedCommit(engine);
  const options = useShapeOptions(engine);
  const pair = useActiveColorPair();
  const colorCommands = useActiveColorCommands();
  const selected = useActiveProjectSelector(
    (project): SelectedShape | null => {
      const { document } = project.canvas;
      const layer = document.selectedLayerId ? getDocumentLayer(document, document.selectedLayerId) : undefined;
      return layer && layer.type === 'raster' && layer.source.type === 'shape'
        ? { id: layer.id, name: layer.name, source: layer.source }
        : null;
    },
    (a, b) => a?.id === b?.id && a?.name === b?.name && a?.source === b?.source
  );
  const fill = selected ? selected.source.fill : options.fillEnabled ? pair.foreground : null;
  const stroke = selected ? selected.source.stroke : options.strokeEnabled ? pair.background : null;
  const strokeWidth = selected ? selected.source.strokeWidth : options.strokeWidth;

  const selectedId = selected?.id ?? null;
  // Scrubbers keep the handler a gesture started with, so writes patch the live source and options: an undo or
  // hotkey landing mid-drag survives the release.
  const commitSource = useCallback(
    (patch: ShapeStylePatch) => {
      if (!selectedId) {
        return;
      }
      commitPrepared(t('widgets.canvas.toolOptions.shapeEdit'), (model) => {
        const layer = model.getLayer(selectedId);
        if (layer?.type !== 'raster' || layer.source.type !== 'shape') {
          return { ids: [selectedId], status: 'missing' };
        }
        return model.prepare({ id: selectedId, source: { ...layer.source, ...patch }, type: 'patch-source' });
      });
    },
    [commitPrepared, selectedId, t]
  );
  const previewStrokeWidth = useCallback((value: number) => setShapeOptions(engine, { strokeWidth: value }), [engine]);
  const setStrokeWidth = useCallback(
    (value: number) => {
      setShapeOptions(engine, { strokeWidth: value });
      commitSource({ strokeWidth: value });
    },
    [commitSource, engine]
  );
  const setFillEnabled = useCallback(
    (checked: boolean) => {
      setShapeOptions(engine, { fillEnabled: checked });
      commitSource({ fill: checked ? (selected?.source.fill ?? pair.foreground) : null });
    },
    [commitSource, engine, pair.foreground, selected]
  );
  const setStrokeEnabled = useCallback(
    (checked: boolean) => {
      setShapeOptions(engine, { strokeEnabled: checked });
      commitSource({ stroke: checked ? (selected?.source.stroke ?? pair.background) : null });
    },
    [commitSource, engine, pair.background, selected]
  );
  const setFillColor = useCallback(
    (hex: string, commit: boolean) => {
      if (selected) {
        if (commit) {
          commitSource({ fill: hex });
        }
        return;
      }
      colorCommands.setPairColor('foreground', hex);
    },
    [colorCommands, commitSource, selected]
  );
  const setStrokeColor = useCallback(
    (hex: string, commit: boolean) => {
      if (selected) {
        if (commit) {
          commitSource({ stroke: hex });
        }
        return;
      }
      colorCommands.setPairColor('background', hex);
    },
    [colorCommands, commitSource, selected]
  );
  return {
    fill,
    previewStrokeWidth,
    selectedName: selected?.name ?? null,
    setFillColor,
    setFillEnabled,
    setStrokeColor,
    setStrokeEnabled,
    setStrokeWidth,
    stroke,
    strokeWidth,
  };
};

/** Fill and stroke: the selected shape layer's own when one is selected (the chip names it), else the defaults. */
const ShapeAppearanceSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const editor = useShapeAppearance(engine);
  const sampleColor = useColorSampler(engine);
  const onFillChange = useCallback((hex: string) => editor.setFillColor(hex, false), [editor]);
  const onFillChangeEnd = useCallback((hex: string) => editor.setFillColor(hex, true), [editor]);
  const onStrokeChange = useCallback((hex: string) => editor.setStrokeColor(hex, false), [editor]);
  const onStrokeChangeEnd = useCallback((hex: string) => editor.setStrokeColor(hex, true), [editor]);
  // Slider ticks preview through the options store; ONE document commit lands on release.
  const previewWidth = useCallback(
    (value: number) => editor.previewStrokeWidth(Math.max(0, Math.round(value))),
    [editor]
  );
  const setWidth = useCallback((value: number) => editor.setStrokeWidth(Math.max(0, Math.round(value))), [editor]);
  const widthGesture = useSliderGesture(Math.round(editor.strokeWidth), setWidth, previewWidth);
  return (
    <>
      <EditTargetChip layerName={editor.selectedName} />
      {/* The chip stays enabled-looking but inert when the slot is off; the toggle owns enablement. */}
      <PropertyControlRow label={t('widgets.canvas.toolOptions.shapeFill')}>
        <ColorPicker
          aria-label={t('widgets.canvas.toolOptions.shapeFill')}
          disabled={editor.fill === null}
          value={editor.fill ?? FALLBACK_COLOR}
          onSampleColor={sampleColor}
          onValueChange={onFillChange}
          onValueChangeEnd={onFillChangeEnd}
        />
        <ToggleIconButton
          checked={editor.fill !== null}
          icon={PaintBucketIcon}
          label={t('widgets.canvas.toolOptions.shapeFill')}
          onCheckedChange={editor.setFillEnabled}
        />
      </PropertyControlRow>
      <PropertyControlRow label={t('widgets.canvas.toolOptions.shapeStroke')}>
        <ColorPicker
          aria-label={t('widgets.canvas.toolOptions.shapeStroke')}
          disabled={editor.stroke === null}
          value={editor.stroke ?? FALLBACK_COLOR}
          onSampleColor={sampleColor}
          onValueChange={onStrokeChange}
          onValueChangeEnd={onStrokeChangeEnd}
        />
        <ToggleIconButton
          checked={editor.stroke !== null}
          icon={SquareIcon}
          label={t('widgets.canvas.toolOptions.shapeStroke')}
          onCheckedChange={editor.setStrokeEnabled}
        />
      </PropertyControlRow>
      <ScrubberField
        defaultValue={DEFAULT_SHAPE_OPTIONS.strokeWidth}
        disabled={editor.stroke === null}
        formatValue={formatPx}
        label={t('widgets.canvas.toolOptions.shapeStrokeWidth')}
        max={MAX_SHAPE_STROKE_WIDTH}
        min={0}
        step={1}
        value={widthGesture.value}
        onChange={widthGesture.onChange}
        onChangeEnd={widthGesture.onChangeEnd}
      />
    </>
  );
};

export const shapeForm: ToolPropertyForm = {
  groups: [
    { body: ShapeToolSettings, id: 'shape', labelKey: 'widgets.properties.groups.shape' },
    { body: ShapeAppearanceSettings, id: 'shape-appearance', labelKey: 'widgets.properties.groups.appearance' },
  ],
  id: 'shape',
  paintsLeaf: true,
};
