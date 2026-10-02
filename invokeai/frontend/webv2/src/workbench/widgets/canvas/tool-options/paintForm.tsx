import type {
  ToolFormProps,
  ToolFooterProps,
  ToolPropertyForm,
  ToolPropertyGroup,
} from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { ColorPicker } from '@platform/ui/ColorPicker';
import { useActiveColorCommands, useActiveColorPair } from '@workbench/widgets/canvas/color-system/useActiveColors';
import { useBrushOptions, useCanvasActiveTool, useEraserOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { PropertyControlRow, PropertySwitchRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import {
  clampBrushSize,
  PaintHardnessControl,
  PaintOpacityControl,
  PaintSizeControl,
  PaintStrokePreview,
} from './BrushOptions';

/** The eraser preview's neutral ink: it erases, so no project color applies. */
const ERASER_PREVIEW_COLOR = '#9aa2b1';

/** Share components/group ids across brush and eraser so switching tools preserves form DOM and geometry. */
const usePaintOptions = (engine: ToolFormProps['engine']) => {
  const activeTool = useCanvasActiveTool(engine);
  const brush = useBrushOptions(engine);
  const eraser = useEraserOptions(engine);
  const isEraser = activeTool === 'eraser';
  const options = isEraser ? eraser : brush;
  const set = useCallback(
    (changes: Partial<typeof options>) => {
      if (isEraser) {
        engine.interaction.set('eraserOptions', { ...eraser, ...changes });
      } else {
        engine.interaction.set('brushOptions', { ...brush, ...changes });
      }
    },
    [brush, engine, eraser, isEraser]
  );
  return { isEraser, options, set };
};

const PaintPreview = ({ engine }: ToolFooterProps) => {
  const { isEraser, options } = usePaintOptions(engine);
  const pair = useActiveColorPair();
  return (
    <PaintStrokePreview
      color={isEraser ? ERASER_PREVIEW_COLOR : pair.foreground}
      hardness={options.hardness}
      opacity={options.opacity}
      size={options.size}
    />
  );
};

const PaintStrokeSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const { isEraser, options, set } = usePaintOptions(engine);
  const setSize = useCallback((size: number) => set({ size: clampBrushSize(size) }), [set]);
  const setOpacity = useCallback((opacity: number) => set({ opacity }), [set]);
  const setHardness = useCallback((hardness: number) => set({ hardness }), [set]);
  return (
    <>
      <PropertyControlRow label={t('widgets.canvas.toolOptions.size')}>
        <PaintSizeControl
          label={t(isEraser ? 'widgets.canvas.toolOptions.eraserSize' : 'widgets.canvas.toolOptions.brushSize')}
          setSize={setSize}
          size={options.size}
        />
      </PropertyControlRow>
      <PropertyControlRow label={t('widgets.canvas.toolOptions.opacity')}>
        <PaintOpacityControl opacity={options.opacity} setOpacity={setOpacity} />
      </PropertyControlRow>
      <PropertyControlRow label={t('widgets.canvas.toolOptions.hardness')}>
        <PaintHardnessControl hardness={options.hardness} setHardness={setHardness} />
      </PropertyControlRow>
    </>
  );
};

/** A mirror of the project foreground, not brush-owned state: the pair feeds the engine's brush color. */
const PaintColorSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const pair = useActiveColorPair();
  const { setPairColor } = useActiveColorCommands();
  const onColorChange = useCallback((color: string) => setPairColor('foreground', color), [setPairColor]);
  const sampleColor = useColorSampler(engine);
  return (
    <PropertyControlRow label={t('widgets.properties.foreground')}>
      <ColorPicker
        aria-label={t('widgets.canvas.toolOptions.brushColor')}
        value={pair.foreground}
        onSampleColor={sampleColor}
        onValueChange={onColorChange}
      />
    </PropertyControlRow>
  );
};

/** Width and opacity are separate pressure responses (opacity also costs a scratch refill per frame). */
const PaintDynamicsSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const brush = useBrushOptions(engine);
  const set = useCallback(
    (changes: Partial<typeof brush>) => engine.interaction.set('brushOptions', { ...brush, ...changes }),
    [brush, engine]
  );
  const onWidth = useCallback((pressureAffectsWidth: boolean) => set({ pressureAffectsWidth }), [set]);
  const onOpacity = useCallback((pressureAffectsOpacity: boolean) => set({ pressureAffectsOpacity }), [set]);
  return (
    <>
      <PropertySwitchRow
        checked={brush.pressureAffectsWidth}
        label={t('widgets.canvas.toolOptions.pressureAffectsWidth')}
        onCheckedChange={onWidth}
      />
      <PropertySwitchRow
        checked={brush.pressureAffectsOpacity}
        label={t('widgets.canvas.toolOptions.pressureAffectsOpacity')}
        onCheckedChange={onOpacity}
      />
    </>
  );
};

/** Shared literal: the same object in both forms, so id, type and DOM survive the switch. */
const STROKE_GROUP: ToolPropertyGroup = {
  body: PaintStrokeSettings,
  id: 'paint-stroke',
  labelKey: 'widgets.properties.groups.stroke',
};

export const brushForm: ToolPropertyForm = {
  groups: [
    STROKE_GROUP,
    { body: PaintColorSettings, id: 'paint-color', labelKey: 'widgets.properties.rows.color' },
    {
      body: PaintDynamicsSettings,
      collapsible: 'collapsed',
      id: 'paint-dynamics',
      labelKey: 'widgets.properties.groups.dynamics',
    },
  ],
  id: 'brush',
  paintsLeaf: true,
  preview: PaintPreview,
};

export const eraserForm: ToolPropertyForm = {
  groups: [STROKE_GROUP],
  id: 'eraser',
  paintsLeaf: true,
  preview: PaintPreview,
};
