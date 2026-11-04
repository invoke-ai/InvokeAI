import type { PaintBucketOptions } from '@workbench/canvas-engine/api';
import type { ToolFormProps, ToolPropertyForm } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { Text } from '@chakra-ui/react';
import { ScrubberField } from '@platform/ui/ScrubberField';
import {
  DEFAULT_PAINT_BUCKET_OPTIONS,
  lookupDocumentLeaf,
  MAX_FILL_CLOSE_GAPS,
  MAX_FILL_EXPAND,
  MAX_FILL_OPACITY,
  MAX_FILL_TOLERANCE,
  paintsInPlace,
} from '@workbench/canvas-engine/api';
import { usePaintBucketOptions, usePaintBucketScrubTolerance } from '@workbench/widgets/canvas/engineStoreHooks';
import { PAINT_COLOR_GROUP } from '@workbench/widgets/canvas/tool-options/paintForm';
import {
  EditTargetChip,
  PropertySegmentedRow,
  PropertySwitchRow,
} from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const formatPercent = (value: number): string => `${value}%`;
const formatPx = (value: number): string => `${value}px`;
const formatSignedPx = (value: number): string => (value > 0 ? `+${value}px` : `${value}px`);
/** No expansion is a stop on the Expand track, whichever side the default sits on. */
const EXPAND_MARKS = [0];

/**
 * What a press fills, by the engine's paint-target rule (`paintsInPlace`): the selected mask (coverage, always hard and opaque),
 * the selected pixel layer in place, or a new paint layer (a group, text, shape or gradient selected, or nothing).
 */
interface BucketTarget {
  kind: 'mask' | 'pixels' | 'new';
  name: string | null;
}

const NEW_LAYER_TARGET: BucketTarget = { kind: 'new', name: null };

const PaintBucketSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const options = usePaintBucketOptions(engine);
  // A held press scrubs tolerance before it becomes the option on release; the row follows the scrub.
  const scrubTolerance = usePaintBucketScrubTolerance(engine);
  const target = useActiveProjectSelector(
    (project): BucketTarget => {
      const { document } = project.canvas;
      const leaf = document.selectedLayerId ? lookupDocumentLeaf(document, document.selectedLayerId) : null;
      if (!leaf || !paintsInPlace(leaf)) {
        return NEW_LAYER_TARGET;
      }
      const { layer } = leaf;
      const isMask = layer.type === 'inpaint_mask' || layer.type === 'regional_guidance';
      return { kind: isMask ? 'mask' : 'pixels', name: layer.name };
    },
    (a, b) => a.kind === b.kind && a.name === b.name
  );
  const isMask = target.kind === 'mask';
  // Patch the store's current options: a scrubber drag keeps the handler it started with, so options captured at
  // its start would undo a tolerance scrubbed on the canvas mid-drag.
  const set = useCallback(
    (changes: Partial<PaintBucketOptions>) =>
      engine.interaction.set('paintBucketOptions', { ...engine.interaction.get('paintBucketOptions'), ...changes }),
    [engine]
  );
  const setMode = useCallback((mode: PaintBucketOptions['mode']) => set({ mode }), [set]);
  const setTolerance = useCallback((tolerance: number) => set({ tolerance }), [set]);
  const setContiguous = useCallback((contiguous: boolean) => set({ contiguous }), [set]);
  const setSample = useCallback((sample: PaintBucketOptions['sample']) => set({ sample }), [set]);
  const setCloseGaps = useCallback((closeGaps: number) => set({ closeGaps }), [set]);
  // Masks and pixels keep separate expansions: a mask wants a small bleed past the edge, a raster fill none.
  const setExpand = useCallback(
    (expand: number) => set(isMask ? { expandMask: expand } : { expandRaster: expand }),
    [isMask, set]
  );
  const setAntiAlias = useCallback((antiAlias: boolean) => set({ antiAlias }), [set]);
  const setOpacity = useCallback((opacity: number) => set({ opacity }), [set]);
  const modeOptions = useMemo(
    () => [
      { label: t('widgets.canvas.toolOptions.paintBucketFill'), value: 'fill' as const },
      { label: t('widgets.canvas.toolOptions.paintBucketErase'), value: 'erase' as const },
    ],
    [t]
  );
  const sampleOptions = useMemo(
    () => [
      { label: t('widgets.canvas.toolOptions.paintBucketSampleAll'), value: 'all' as const },
      { label: t('widgets.canvas.toolOptions.paintBucketSampleLayer'), value: 'layer' as const },
    ],
    [t]
  );
  // A fill with nothing that takes pixels creates a layer; an erase has nothing to erase and is refused.
  const erasesNothing = target.kind === 'new' && options.mode === 'erase';
  return (
    <>
      {erasesNothing ? (
        <Text color="fg.muted" fontSize="md" minW="0">
          {t('widgets.canvas.toolOptions.paintBucketEraseNeedsLayer')}
        </Text>
      ) : (
        <EditTargetChip layerName={target.name ?? t('widgets.properties.target.newLayer')} />
      )}
      <PropertySegmentedRow
        label={t('widgets.properties.rows.mode')}
        options={modeOptions}
        value={options.mode}
        onValueChange={setMode}
      />
      <ScrubberField
        defaultValue={DEFAULT_PAINT_BUCKET_OPTIONS.tolerance}
        formatValue={formatPercent}
        label={t('widgets.canvas.toolOptions.paintBucketTolerance')}
        max={MAX_FILL_TOLERANCE}
        min={0}
        step={1}
        value={scrubTolerance ?? options.tolerance}
        onChange={setTolerance}
      />
      <PropertySwitchRow
        checked={options.contiguous}
        label={t('widgets.canvas.toolOptions.paintBucketContiguous')}
        onCheckedChange={setContiguous}
      />
      <PropertySegmentedRow
        label={t('widgets.canvas.toolOptions.paintBucketSample')}
        options={sampleOptions}
        value={options.sample}
        onValueChange={setSample}
      />
      {/* Gaps close along the connected region only, so the row is inert while Contiguous is off. */}
      <ScrubberField
        defaultValue={DEFAULT_PAINT_BUCKET_OPTIONS.closeGaps}
        disabled={!options.contiguous}
        formatValue={formatPx}
        helpText={t('widgets.canvas.toolOptions.paintBucketCloseGapsHelp')}
        label={t('widgets.canvas.toolOptions.paintBucketCloseGaps')}
        max={MAX_FILL_CLOSE_GAPS}
        min={0}
        step={1}
        value={options.closeGaps}
        onChange={setCloseGaps}
      />
      <ScrubberField
        defaultValue={isMask ? DEFAULT_PAINT_BUCKET_OPTIONS.expandMask : DEFAULT_PAINT_BUCKET_OPTIONS.expandRaster}
        formatValue={formatSignedPx}
        label={t('widgets.canvas.toolOptions.paintBucketExpand')}
        marks={EXPAND_MARKS}
        max={MAX_FILL_EXPAND}
        min={-MAX_FILL_EXPAND}
        step={1}
        value={isMask ? options.expandMask : options.expandRaster}
        onChange={setExpand}
      />
      {/* A mask fill is always hard and opaque, so its edge and strength rows are hidden rather than inert. */}
      {isMask ? null : (
        <>
          <PropertySwitchRow
            checked={options.antiAlias}
            label={t('widgets.canvas.toolOptions.paintBucketAntiAlias')}
            onCheckedChange={setAntiAlias}
          />
          <ScrubberField
            defaultValue={DEFAULT_PAINT_BUCKET_OPTIONS.opacity}
            formatValue={formatPercent}
            label={t('widgets.canvas.toolOptions.opacity')}
            max={MAX_FILL_OPACITY}
            min={0}
            step={1}
            value={options.opacity}
            onChange={setOpacity}
          />
        </>
      )}
    </>
  );
};

/**
 * Not `paintsLeaf`: with a group selected the bucket fills a new layer above it, so the pane's "select a layer
 * inside this group" notice would be wrong; the edit-target chip names the new layer instead.
 */
export const paintBucketForm: ToolPropertyForm = {
  groups: [
    { body: PaintBucketSettings, id: 'paint-bucket', labelKey: 'widgets.properties.groups.settings' },
    PAINT_COLOR_GROUP,
  ],
  id: 'paintBucket',
};
