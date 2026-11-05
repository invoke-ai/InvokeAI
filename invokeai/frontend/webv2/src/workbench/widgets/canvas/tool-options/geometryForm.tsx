import type { AspectRatioId } from '@features/generation/contracts';
import type { CanvasLayerContract, LayerTransform, TransformOptions } from '@workbench/canvas-engine/api';
import type {
  ToolFormProps,
  ToolPropertyForm,
  ToolPropertyGroup,
} from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { createListCollection, Flex, Stack, Text } from '@chakra-ui/react';
import { AspectRatioLockButton, AspectRatioSelect } from '@features/generation/components';
import { ASPECT_RATIO_MAP, deriveAspectRatioId } from '@features/generation/settings';
import { Button } from '@platform/ui/Button';
import { Select } from '@platform/ui/Select';
import { Tooltip } from '@platform/ui/Tooltip';
import { constrainBboxToRatio, lookupDocumentLeaf } from '@workbench/canvas-engine/api';
import {
  useCanvasActiveTool,
  useCanvasDocumentEditingLocked,
  useCanvasHasFloatingSelection,
  usePendingTransformLayerId,
  useTransformOptions,
  useTransformSession,
} from '@workbench/widgets/canvas/engineStoreHooks';
import {
  FormNumberField,
  ApplyCancelBar,
  useNumberCommit,
} from '@workbench/widgets/canvas/tool-presentation/FormControls';
import { PropertyControlRow, PropertySwitchRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { fitLayerTransformToBbox } from '@workbench/widgets/layers/layerOps';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useBboxEditor } from './useBboxEditor';

const RAD_TO_DEG = 180 / Math.PI;
const DEG_TO_RAD = Math.PI / 180;
const MIN_SCALE_PERCENT = 1;
const ASPECT_TRIGGER_PROPS = { minW: '7.5rem' } as const;

export const round2 = (n: number): number => Math.round(n * 100) / 100;
/** Degrees in (-180, 180], so 270 reads and stores as -90. */
export const wrapDegrees = (degrees: number): number => {
  const wrapped = ((((degrees + 180) % 360) + 360) % 360) - 180;
  return wrapped === -180 ? 180 : wrapped;
};
/** A flip is a negative scale; a zero or near-zero one is not a transform anyone can undo by eye. */
export const clampScalePercent = (percent: number): number =>
  percent < 0 ? Math.min(percent, -MIN_SCALE_PERCENT) : Math.max(percent, MIN_SCALE_PERCENT);

interface SelectedLeafTransform {
  id: string;
  locked: boolean;
  transform: LayerTransform;
}

/**
 * The one layer-transform write path the Move and Transform tools (and the
 * Transform pane) share: a live transform session absorbs edits into its
 * single Apply commit; otherwise each settled field is one undoable patch on
 * the selected, unlocked leaf.
 */
export const useLayerTransformEditor = (engine: ToolFormProps['engine'], labelKey = 'widgets.transform.edit') => {
  const { t } = useTranslation();
  const commitPrepared = usePreparedCommit(engine);
  const session = useTransformSession(engine);
  const activeTool = useCanvasActiveTool(engine);
  const hasFloat = useCanvasHasFloatingSelection(engine);
  const locked = useCanvasDocumentEditingLocked(engine);
  const selected = useActiveProjectSelector(
    (project): SelectedLeafTransform | null => {
      const { document } = project.canvas;
      const leaf = document.selectedLayerId ? lookupDocumentLeaf(document, document.selectedLayerId) : null;
      return leaf ? { id: leaf.id, locked: leaf.effectiveLocked, transform: leaf.layer.transform } : null;
    },
    (a, b) => a?.id === b?.id && a?.locked === b?.locked && a?.transform === b?.transform
  );
  const transform = session?.transform ?? (selected && !selected.locked ? selected.transform : null);
  const editable =
    !locked &&
    !session?.processing &&
    selected !== null &&
    !selected.locked &&
    (activeTool !== 'transform' || hasFloat || session?.layerId === selected.id);
  const patch = useCallback(
    (next: Partial<LayerTransform>) => {
      if (!editable) {
        return;
      }
      if (session) {
        engine.layers.updateTransformSession({ ...session.transform, ...next });
        return;
      }
      if (!selected || selected.locked) {
        return;
      }
      commitPrepared(t(labelKey), (model) =>
        model.prepare({ id: selected.id, patch: { transform: next }, type: 'patch' })
      );
    },
    [commitPrepared, editable, engine, labelKey, selected, session, t]
  );
  return { editable, patch, transform };
};

/** Share geometry rows across frame, selected layer, and transform session to preserve X/Y DOM identity. */
const GeometryPositionSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const activeTool = useCanvasActiveTool(engine);
  const isFrame = activeTool === 'bbox';
  const isTransform = activeTool === 'transform';
  // The move tool keeps its own undo verbiage; everything else is a transform edit.
  const layerEditor = useLayerTransformEditor(
    engine,
    activeTool === 'move' ? 'widgets.canvas.toolOptions.movePosition' : undefined
  );
  const bboxEditor = useBboxEditor(engine);
  const x = isFrame ? bboxEditor.bbox.x : layerEditor.transform ? Math.round(layerEditor.transform.x) : null;
  const y = isFrame ? bboxEditor.bbox.y : layerEditor.transform ? Math.round(layerEditor.transform.y) : null;
  const onX = useNumberCommit(
    useCallback(
      (value: number) => (isFrame ? bboxEditor.setX(value) : layerEditor.patch({ x: Math.round(value) })),
      [bboxEditor, isFrame, layerEditor]
    )
  );
  const onY = useNumberCommit(
    useCallback(
      (value: number) => (isFrame ? bboxEditor.setY(value) : layerEditor.patch({ y: Math.round(value) })),
      [bboxEditor, isFrame, layerEditor]
    )
  );
  return (
    <Flex align={isTransform ? 'center' : undefined} gap="2" w="full">
      <Text color="fg.muted" flexShrink={0} hidden={!isTransform} w="6rem">
        {t('widgets.transform.position')}
      </Text>
      <FormNumberField
        aria-label={t('widgets.canvas.toolOptions.positionX')}
        disabled={x === null || (!isFrame && !layerEditor.editable)}
        label={t('widgets.canvas.toolOptions.positionX')}
        value={x === null ? '' : String(x)}
        onValueCommit={onX}
      />
      <FormNumberField
        aria-label={t('widgets.canvas.toolOptions.positionY')}
        disabled={y === null || (!isFrame && !layerEditor.editable)}
        label={t('widgets.canvas.toolOptions.positionY')}
        value={y === null ? '' : String(y)}
        onValueCommit={onY}
      />
    </Flex>
  );
};

const RESAMPLE_MODES = createListCollection({
  items: [
    { label: 'Bilinear', value: 'bilinear' },
    { label: 'Bicubic', value: 'bicubic' },
    { label: 'Hamming', value: 'hamming' },
    { label: 'Lanczos', value: 'lanczos' },
  ],
});

/** Transform-tool options act on its existing preview session, never on the layer-properties write path. */
const TransformResamplingSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const session = useTransformSession(engine);
  const options = useTransformOptions(engine);
  const locked = useCanvasDocumentEditingLocked(engine);
  const hasFloat = useCanvasHasFloatingSelection(engine);
  const [error, setError] = useState<{ layerId: string; message: string } | null>(null);
  const selected = useActiveProjectSelector(
    (project): { layer: CanvasLayerContract; locked: boolean } | null => {
      const document = project.canvas.document;
      const leaf = document.selectedLayerId ? lookupDocumentLeaf(document, document.selectedLayerId) : null;
      return leaf ? { layer: leaf.layer, locked: leaf.effectiveLocked } : null;
    },
    (a, b) => a?.layer === b?.layer && a?.locked === b?.locked
  );
  const editable =
    !!session &&
    !!selected &&
    session.layerId === selected.layer.id &&
    !session.processing &&
    !selected.locked &&
    !locked &&
    !hasFloat;
  const pixelSource =
    selected &&
    (selected.layer.type === 'raster' || selected.layer.type === 'control') &&
    (selected.layer.source.type === 'image' || selected.layer.source.type === 'paint');
  const smoothingAvailable = editable && !!pixelSource;
  const fitCollection = useMemo(
    () =>
      createListCollection({
        items: [
          { label: t('widgets.transform.contain'), value: 'contain' },
          { label: t('widgets.transform.cover'), value: 'cover' },
          { label: t('widgets.transform.fill'), value: 'fill' },
        ],
      }),
    [t]
  );
  const resampleValue = useMemo(() => [options.resampleMode], [options.resampleMode]);
  const fitValue = useMemo(() => [options.fitMode], [options.fitMode]);
  const setSmoothing = useCallback(
    (smoothing: boolean) => engine.interaction.set('transformOptions', { ...options, smoothing }),
    [engine, options]
  );
  const setResampleMode = useCallback(
    ({ value }: { value: string[] }) => {
      if (value[0]) {
        engine.interaction.set('transformOptions', {
          ...options,
          resampleMode: value[0] as TransformOptions['resampleMode'],
        });
      }
    },
    [engine, options]
  );
  const setFitMode = useCallback(
    ({ value }: { value: string[] }) => {
      if (value[0]) {
        engine.interaction.set('transformOptions', { ...options, fitMode: value[0] as TransformOptions['fitMode'] });
      }
    },
    [engine, options]
  );
  const onFit = useCallback(() => {
    const document = engine.document.getDocument();
    if (!editable || !selected || !document || !session) {
      return;
    }
    const transform = fitLayerTransformToBbox(
      selected.layer,
      document.bbox,
      { ...document.bbox, width: document.width, height: document.height },
      options.fitMode
    );
    if (!transform) {
      setError({ layerId: selected.layer.id, message: t('widgets.canvas.structural.notReady') });
      return;
    }
    setError(null);
    engine.layers.updateTransformSession(transform);
  }, [editable, engine, options.fitMode, selected, session, t]);
  return (
    <>
      {error?.layerId === session?.layerId && error ? (
        <Text role="alert" color="fg.error">
          {error.message}
        </Text>
      ) : null}
      <PropertySwitchRow
        checked={options.smoothing}
        disabled={!smoothingAvailable}
        label={t('widgets.transform.smoothing')}
        onCheckedChange={setSmoothing}
      />
      <PropertyControlRow label={t('widgets.transform.resampleMode')} labelWidth="6rem">
        <Select
          aria-label={t('widgets.transform.resampleMode')}
          collection={RESAMPLE_MODES}
          disabled={!smoothingAvailable || !options.smoothing}
          gridColumn="2 / -1"
          value={resampleValue}
          onValueChange={setResampleMode}
        />
      </PropertyControlRow>
      <PropertyControlRow label={t('widgets.transform.fitMode')} labelWidth="6rem">
        <Flex gap="2" gridColumn="2 / -1" minW="0">
          <Select
            aria-label={t('widgets.transform.fitMode')}
            collection={fitCollection}
            disabled={!editable}
            flex="1"
            minW="0"
            value={fitValue}
            onValueChange={setFitMode}
          />
          <Button disabled={!editable} onClick={onFit}>
            {t('widgets.transform.fitToBbox')}
          </Button>
        </Flex>
      </PropertyControlRow>
      {options.smoothing ? (
        <Text color="fg.muted" fontSize="xs">
          {t('widgets.transform.smoothingHint')}
        </Text>
      ) : null}
    </>
  );
};

/** Scale percent pair and rotation for the transform session / selected leaf. */
export const TransformScaleSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const { editable, patch, transform } = useLayerTransformEditor(engine);
  const onScaleX = useNumberCommit(
    useCallback((value: number) => patch({ scaleX: clampScalePercent(value) / 100 }), [patch])
  );
  const onScaleY = useNumberCommit(
    useCallback((value: number) => patch({ scaleY: clampScalePercent(value) / 100 }), [patch])
  );
  const onRotation = useNumberCommit(
    useCallback((value: number) => patch({ rotation: wrapDegrees(value) * DEG_TO_RAD }), [patch])
  );
  return (
    <Stack gap="2">
      <PropertyControlRow label={t('widgets.transform.scale')} labelWidth="6rem">
        <Flex gap="2" gridColumn="2 / -1">
          <FormNumberField
            aria-label={t('widgets.canvas.toolOptions.scaleWidth')}
            disabled={!editable}
            label={t('widgets.canvas.toolOptions.frameWidth')}
            suffix="%"
            value={transform ? String(round2(transform.scaleX * 100)) : ''}
            onValueCommit={onScaleX}
          />
          <FormNumberField
            aria-label={t('widgets.canvas.toolOptions.scaleHeight')}
            disabled={!editable}
            label={t('widgets.canvas.toolOptions.frameHeight')}
            suffix="%"
            value={transform ? String(round2(transform.scaleY * 100)) : ''}
            onValueCommit={onScaleY}
          />
        </Flex>
      </PropertyControlRow>
      <PropertyControlRow label={t('widgets.transform.rotation')} labelWidth="6rem">
        <FormNumberField
          aria-label={t('widgets.canvas.toolOptions.rotation')}
          disabled={!editable}
          suffix="°"
          value={transform ? String(round2(wrapDegrees(transform.rotation * RAD_TO_DEG))) : ''}
          onValueCommit={onRotation}
        />
      </PropertyControlRow>
    </Stack>
  );
};

/**
 * Frame width/height plus the aspect preset and lock, as form rows. While the canvas drives Generate the lock is
 * Generate's own, one lock for both (the Frame tool follows it, see `CanvasFrameLockFeed`); otherwise it is the
 * Frame tool's alone.
 */
const FrameSizeSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const { generation } = useWorkbenchCommands();
  const isLinked = useActiveProjectSelector((project) => project.invocation.sourceId === 'canvas');
  const { bbox, commitBbox, grid, options, setHeight, setWidth } = useBboxEditor(engine);
  const onWidth = useNumberCommit(setWidth);
  const onHeight = useNumberCommit(setHeight);
  const bboxRatio = bbox.height > 0 ? bbox.width / bbox.height : 1;
  // Derived from the live frame so a lock captured from a hand-drawn bbox reports the preset it matches.
  const selectedId: AspectRatioId = options.aspectLocked ? deriveAspectRatioId(bbox.width, bbox.height) : 'Free';

  const setLock = useCallback(
    (aspectLocked: boolean, aspectRatio: number) => {
      if (isLinked) {
        // Generate's fields; the feed brings the lock back to the Frame tool.
        generation.patchSettings({
          aspectRatioId: aspectLocked ? deriveAspectRatioId(bbox.width, bbox.height) : 'Free',
          aspectRatioIsLocked: aspectLocked,
          aspectRatioValue: aspectRatio,
        });
      }
      engine.interaction.set('bboxOptions', { aspectLocked, aspectRatio });
    },
    [bbox.height, bbox.width, engine, generation, isLinked]
  );

  const onAspectPresetChange = useCallback(
    (id: AspectRatioId) => {
      if (id === 'Free') {
        setLock(false, bboxRatio);
        return;
      }
      const ratio = ASPECT_RATIO_MAP[id].ratio;
      if (isLinked) {
        generation.patchSettings({ aspectRatioId: id, aspectRatioIsLocked: true, aspectRatioValue: ratio });
      }
      engine.interaction.set('bboxOptions', { aspectLocked: true, aspectRatio: ratio });
      commitBbox(constrainBboxToRatio(bbox, ratio, grid));
    },
    [bbox, bboxRatio, commitBbox, engine, generation, grid, isLinked, setLock]
  );

  const onLockToggle = useCallback(() => {
    const checked = !options.aspectLocked;
    // Locking a frame that matches no preset captures its current ratio.
    const aspectRatio =
      checked && bbox.height > 0 && deriveAspectRatioId(bbox.width, bbox.height) === 'Free'
        ? bboxRatio
        : options.aspectRatio > 0
          ? options.aspectRatio
          : 1;
    setLock(checked, checked ? aspectRatio : bboxRatio);
  }, [bbox, bboxRatio, options.aspectLocked, options.aspectRatio, setLock]);

  return (
    <>
      <PropertyControlRow label={t('widgets.properties.rows.size')}>
        <Flex gap="2" gridColumn="2 / -1">
          <FormNumberField
            aria-label={t('widgets.canvas.toolOptions.frameWidth')}
            label={t('widgets.canvas.toolOptions.frameWidth')}
            min={1}
            value={String(bbox.width)}
            onValueCommit={onWidth}
          />
          <FormNumberField
            aria-label={t('widgets.canvas.toolOptions.frameHeight')}
            label={t('widgets.canvas.toolOptions.frameHeight')}
            min={1}
            value={String(bbox.height)}
            onValueCommit={onHeight}
          />
        </Flex>
      </PropertyControlRow>
      <PropertyControlRow label={t('widgets.properties.rows.aspect')}>
        <Flex align="center" gap="2" gridColumn="2 / -1">
          <AspectRatioSelect
            fallbackRatio={bboxRatio}
            triggerProps={ASPECT_TRIGGER_PROPS}
            value={selectedId}
            onChange={onAspectPresetChange}
          />
          <AspectRatioLockButton isLocked={options.aspectLocked} onToggle={onLockToggle} />
        </Flex>
      </PropertyControlRow>
      {isLinked ? (
        <Text color="fg.muted" fontSize="xs" gridColumn="1 / -1">
          {t('widgets.canvas.toolOptions.frameLinked')}
        </Text>
      ) : null}
    </>
  );
};

/**
 * Apply / Cancel stay live for a floating selection, which frames its own
 * pixels instead of opening a layer session; the numerics meanwhile edit the
 * settled selected layer, matching the Transform pane.
 */
const TransformFooter = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const session = useTransformSession(engine);
  const hasFloat = useCanvasHasFloatingSelection(engine);
  const isPending = usePendingTransformLayerId(engine) !== null;
  const disabled = !session && !hasFloat;
  const onApply = useCallback(() => engine.layers.applyTransform(), [engine]);
  const onCancel = useCallback(() => engine.layers.cancelTransform(), [engine]);
  const pendingText = t('widgets.canvas.transformPending');
  return (
    <ApplyCancelBar
      applyDisabled={disabled || session?.processing === true}
      applyLoading={session?.processing}
      cancelDisabled={disabled}
      onApply={onApply}
      onCancel={onCancel}
    >
      {/* Always mounted, so screen readers hear the wait start and end as the live region's text changes. The
          tooltip shows the whole line where a narrow pane truncates it. */}
      <Text color="fg.muted" role="status" truncate>
        {isPending ? (
          <Tooltip content={pendingText}>
            <span>{pendingText}</span>
          </Tooltip>
        ) : null}
      </Text>
    </ApplyCancelBar>
  );
};

/** Shared literal: the same object in every geometry form, so the rows' DOM survives tool switches. */
const POSITION_GROUP: ToolPropertyGroup = {
  body: GeometryPositionSettings,
  id: 'geometry-position',
  labelKey: 'widgets.transform.position',
};

export const moveForm: ToolPropertyForm = {
  groups: [POSITION_GROUP],
  id: 'move',
};

export const bboxForm: ToolPropertyForm = {
  groups: [POSITION_GROUP, { body: FrameSizeSettings, id: 'frame-size', labelKey: 'widgets.properties.groups.frame' }],
  id: 'bbox',
};

export const transformForm: ToolPropertyForm = {
  footer: TransformFooter,
  groups: [
    { body: TransformResamplingSettings, id: 'transform-resampling', labelKey: 'widgets.transform.resampling' },
    { ...POSITION_GROUP, hideLabel: true },
    { body: TransformScaleSettings, id: 'transform-scale', labelKey: 'widgets.transform.scale', hideLabel: true },
  ],
  id: 'transform',
};
