import type {
  CanvasLayerContract,
  CanvasLayerSourceContract,
  ShapeToolKind,
  ShapeToolOptions,
} from '@workbench/canvas-engine/api';
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
import { useColorSampler, type ColorSamplerEngine } from '@workbench/widgets/canvas/useColorSampler';
import { usePreparedCommit, type CanvasPreparedEngine } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { PaintBucketIcon, SquareIcon } from 'lucide-react';
import { useCallback, useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;
type ShapeStylePatch = Partial<Pick<ShapeSource, 'fill' | 'stroke' | 'strokeWidth'>>;

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
  const canPlaceSelected = placement === 'layer' || placement === 'mask';
  const targetOptions = useMemo(
    () => [
      {
        disabled: !canPlaceSelected,
        label: t('widgets.canvas.toolOptions.shapeTargetSelected'),
        value: 'selected' as const,
      },
      { label: t('widgets.canvas.toolOptions.shapeTargetNew'), value: 'new' as const },
    ],
    [canPlaceSelected, t]
  );
  const onTargetChange = useCallback(
    (target: ShapeToolOptions['target']) => setShapeOptions(engine, { target }),
    [engine]
  );
  const hintId = useId();
  const hintKey = !canPlaceSelected
    ? 'widgets.canvas.toolOptions.shapeTargetFallbackHint'
    : options.target === 'selected' && placement === 'mask'
      ? 'widgets.canvas.toolOptions.shapeMaskHint'
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
        describedBy={hintKey ? hintId : undefined}
        label={t('widgets.canvas.toolOptions.shapeTarget')}
        options={targetOptions}
        value={canPlaceSelected ? options.target : 'new'}
        onValueChange={onTargetChange}
      />
      {hintKey ? (
        <Text color="fg.muted" fontSize="xs" id={hintId}>
          {t(hintKey)}
        </Text>
      ) : null}
    </>
  );
};

/** Creation defaults stay independent of the selected shape's appearance. */
const ShapeDefaultAppearanceSettings = ({ engine }: ToolFormProps) => {
  const options = useShapeOptions(engine);
  const pair = useActiveColorPair();
  const colorCommands = useActiveColorCommands();
  const setStrokeWidth = useCallback((value: number) => setShapeOptions(engine, { strokeWidth: value }), [engine]);
  const setFillEnabled = useCallback((checked: boolean) => setShapeOptions(engine, { fillEnabled: checked }), [engine]);
  const setStrokeEnabled = useCallback(
    (checked: boolean) => setShapeOptions(engine, { strokeEnabled: checked }),
    [engine]
  );
  const setFillColor = useCallback((hex: string) => colorCommands.setPairColor('foreground', hex), [colorCommands]);
  const setStrokeColor = useCallback((hex: string) => colorCommands.setPairColor('background', hex), [colorCommands]);
  return (
    <>
      <EditTargetChip layerName={null} />
      <ShapeAppearanceControls
        engine={engine}
        fill={options.fillEnabled ? pair.foreground : null}
        previewStrokeWidth={setStrokeWidth}
        setFillColor={setFillColor}
        setFillEnabled={setFillEnabled}
        setStrokeColor={setStrokeColor}
        setStrokeEnabled={setStrokeEnabled}
        setStrokeWidth={setStrokeWidth}
        stroke={options.strokeEnabled ? pair.background : null}
        strokeWidth={options.strokeWidth}
      />
    </>
  );
};

/** Selected-layer appearance belongs to Layer properties, whichever canvas tool is active. */
export const SelectedShapeAppearanceSettings = ({
  engine,
  layer,
  source,
}: {
  engine: (CanvasPreparedEngine & ColorSamplerEngine) | null;
  layer: Pick<CanvasLayerContract, 'id' | 'name'>;
  source: ShapeSource;
}) => {
  const { t } = useTranslation();
  const commitPrepared = usePreparedCommit(engine);
  const pair = useActiveColorPair();
  // A scrub keeps its starting callback. Patch the live source, and drop a release after selection changed.
  const commitSource = useCallback(
    (patch: ShapeStylePatch) => {
      commitPrepared(t('widgets.canvas.toolOptions.shapeEdit'), (model) => {
        if (model.document.selectedLayerId !== layer.id) {
          return { status: 'unchanged' };
        }
        const current = model.getLayer(layer.id);
        if (current?.type !== 'raster' || current.source.type !== 'shape') {
          return { ids: [layer.id], status: 'missing' };
        }
        return model.prepare({ id: layer.id, source: { ...current.source, ...patch }, type: 'patch-source' });
      });
    },
    [commitPrepared, layer.id, t]
  );
  const setFillEnabled = useCallback(
    (checked: boolean) => commitSource({ fill: checked ? (source.fill ?? pair.foreground) : null }),
    [commitSource, pair.foreground, source.fill]
  );
  const setStrokeEnabled = useCallback(
    (checked: boolean) => commitSource({ stroke: checked ? (source.stroke ?? pair.background) : null }),
    [commitSource, pair.background, source.stroke]
  );
  const setStrokeWidth = useCallback((value: number) => commitSource({ strokeWidth: value }), [commitSource]);
  const setFillColor = useCallback(
    (hex: string, commit: boolean) => {
      if (commit) {
        commitSource({ fill: hex });
      }
    },
    [commitSource]
  );
  const setStrokeColor = useCallback(
    (hex: string, commit: boolean) => {
      if (commit) {
        commitSource({ stroke: hex });
      }
    },
    [commitSource]
  );
  return (
    <ShapeAppearanceControls
      engine={engine}
      fill={source.fill}
      setFillColor={setFillColor}
      setFillEnabled={setFillEnabled}
      setStrokeColor={setStrokeColor}
      setStrokeEnabled={setStrokeEnabled}
      setStrokeWidth={setStrokeWidth}
      stroke={source.stroke}
      strokeWidth={source.strokeWidth}
    />
  );
};

/** The same fill, stroke and width controls for defaults and selected content, with separate write paths. */
const ShapeAppearanceControls = ({
  engine,
  ...editor
}: {
  engine: ColorSamplerEngine | null;
  fill: string | null;
  stroke: string | null;
  strokeWidth: number;
  setFillColor: (hex: string, commit: boolean) => void;
  setStrokeColor: (hex: string, commit: boolean) => void;
  setFillEnabled: (checked: boolean) => void;
  setStrokeEnabled: (checked: boolean) => void;
  setStrokeWidth: (value: number) => void;
  previewStrokeWidth?: (value: number) => void;
}) => {
  const { t } = useTranslation();
  const sampleColor = useColorSampler(engine);
  const onFillChange = useCallback((hex: string) => editor.setFillColor(hex, false), [editor]);
  const onFillChangeEnd = useCallback((hex: string) => editor.setFillColor(hex, true), [editor]);
  const onStrokeChange = useCallback((hex: string) => editor.setStrokeColor(hex, false), [editor]);
  const onStrokeChangeEnd = useCallback((hex: string) => editor.setStrokeColor(hex, true), [editor]);
  const previewWidth = useCallback(
    (value: number) => editor.previewStrokeWidth?.(Math.max(0, Math.round(value))),
    [editor]
  );
  const setWidth = useCallback((value: number) => editor.setStrokeWidth(Math.max(0, Math.round(value))), [editor]);
  const widthGesture = useSliderGesture(Math.round(editor.strokeWidth), setWidth, previewWidth);
  return (
    <>
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
    { body: ShapeDefaultAppearanceSettings, id: 'shape-appearance', labelKey: 'widgets.properties.groups.appearance' },
  ],
  id: 'shape',
  paintsLeaf: true,
};
