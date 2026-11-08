import type { LayerTransform } from '@workbench/canvas-engine/api';
import type {
  ToolFormProps,
  ToolPropertyForm,
  ToolPropertyGroup,
} from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { Flex, Text } from '@chakra-ui/react';
import { AspectRatioLockButton, AspectRatioPicker } from '@features/generation/components';
import { getLockedAspectRatioSettings } from '@features/generation/settings';
import { Tooltip } from '@platform/ui/Tooltip';
import { bboxMinSize, lookupDocumentLeaf } from '@workbench/canvas-engine/api';
import { useAspectRatioPresets } from '@workbench/settings/aspectRatioPresets';
import { getBboxSizeForRatio, resizeBboxAroundCentre } from '@workbench/widgets/canvas/bboxGrid';
import {
  useCanvasActiveTool,
  useCanvasHasFloatingSelection,
  usePendingTransformLayerId,
  useTransformSession,
} from '@workbench/widgets/canvas/engineStoreHooks';
import {
  FormNumberField,
  ApplyCancelBar,
  useNumberCommit,
} from '@workbench/widgets/canvas/tool-presentation/FormControls';
import { PropertyControlRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useBboxEditor } from './useBboxEditor';

const RAD_TO_DEG = 180 / Math.PI;
const DEG_TO_RAD = Math.PI / 180;
const MIN_SCALE_PERCENT = 1;

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
  const selected = useActiveProjectSelector(
    (project): SelectedLeafTransform | null => {
      const { document } = project.canvas;
      const leaf = document.selectedLayerId ? lookupDocumentLeaf(document, document.selectedLayerId) : null;
      return leaf ? { id: leaf.id, locked: leaf.effectiveLocked, transform: leaf.layer.transform } : null;
    },
    (a, b) => a?.id === b?.id && a?.locked === b?.locked && a?.transform === b?.transform
  );
  const transform = session?.transform ?? (selected && !selected.locked ? selected.transform : null);
  const patch = useCallback(
    (next: Partial<LayerTransform>) => {
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
    [commitPrepared, engine, labelKey, selected, session, t]
  );
  return { patch, transform };
};

/** Share geometry rows across frame, selected layer, and transform session to preserve X/Y DOM identity. */
const GeometryPositionSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const activeTool = useCanvasActiveTool(engine);
  const isFrame = activeTool === 'bbox';
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
    <Flex gap="2" w="full">
      <FormNumberField
        aria-label={t('widgets.canvas.toolOptions.positionX')}
        disabled={x === null}
        label={t('widgets.canvas.toolOptions.positionX')}
        value={x === null ? '' : String(x)}
        onValueCommit={onX}
      />
      <FormNumberField
        aria-label={t('widgets.canvas.toolOptions.positionY')}
        disabled={y === null}
        label={t('widgets.canvas.toolOptions.positionY')}
        value={y === null ? '' : String(y)}
        onValueCommit={onY}
      />
    </Flex>
  );
};

/** Scale percent pair and rotation for the transform session / selected leaf. */
export const TransformScaleSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const { patch, transform } = useLayerTransformEditor(engine);
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
    <>
      <PropertyControlRow label={t('widgets.transform.scale')}>
        <Flex gap="2" gridColumn="2 / -1">
          <FormNumberField
            aria-label={t('widgets.canvas.toolOptions.scaleWidth')}
            disabled={!transform}
            label={t('widgets.canvas.toolOptions.frameWidth')}
            suffix="%"
            value={transform ? String(round2(transform.scaleX * 100)) : ''}
            onValueCommit={onScaleX}
          />
          <FormNumberField
            aria-label={t('widgets.canvas.toolOptions.scaleHeight')}
            disabled={!transform}
            label={t('widgets.canvas.toolOptions.frameHeight')}
            suffix="%"
            value={transform ? String(round2(transform.scaleY * 100)) : ''}
            onValueCommit={onScaleY}
          />
        </Flex>
      </PropertyControlRow>
      <PropertyControlRow label={t('widgets.transform.rotation')}>
        <FormNumberField
          aria-label={t('widgets.canvas.toolOptions.rotation')}
          disabled={!transform}
          suffix="°"
          value={transform ? String(round2(wrapDegrees(transform.rotation * RAD_TO_DEG))) : ''}
          onValueCommit={onRotation}
        />
      </PropertyControlRow>
    </>
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
  const aspectRatioPresets = useAspectRatioPresets();
  const isLinked = useActiveProjectSelector((project) => project.invocation.sourceId === 'canvas');
  const { bbox, commitBbox, grid, options, setHeight, setWidth } = useBboxEditor(engine);
  const onWidth = useNumberCommit(setWidth);
  const onHeight = useNumberCommit(setHeight);
  const bboxRatio = bbox.height > 0 ? bbox.width / bbox.height : 1;
  // The lock's own ratio, not the frame's: a grid-snapped frame is only close to the ratio it holds.
  const lockedRatio = options.aspectLocked ? (options.aspectRatio > 0 ? options.aspectRatio : bboxRatio) : null;
  const currentSize = useMemo(() => ({ height: bbox.height, width: bbox.width }), [bbox.height, bbox.width]);

  /** Locks the Frame tool (and Generate, while linked) to `aspectRatio`, or unlocks; returns the ratio stored. */
  const applyLock = useCallback(
    (aspectRatio: number | null): number | null => {
      const locked = aspectRatio === null ? null : getLockedAspectRatioSettings(aspectRatio);
      if (isLinked) {
        // Generate's fields; the feed brings the lock back to the Frame tool.
        generation.patchSettings(
          locked ?? { aspectRatioId: 'Free', aspectRatioIsLocked: false, aspectRatioValue: bboxRatio }
        );
      }
      engine.interaction.set('bboxOptions', {
        aspectLocked: locked !== null,
        aspectRatio: locked?.aspectRatioValue ?? bboxRatio,
      });
      return locked?.aspectRatioValue ?? null;
    },
    [bboxRatio, engine, generation, isLinked]
  );

  // The size a ratio sets keeps the frame's pixel count on its grid, as choosing it in Generate does, without
  // Generate's maximum: the frame may be larger than any one generation.
  const getSizeForRatio = useCallback(
    (ratio: number) => getBboxSizeForRatio(bbox.width * bbox.height, ratio, grid, bboxMinSize(grid)),
    [bbox.height, bbox.width, grid]
  );

  const onAspectRatioChange = useCallback(
    (ratio: number | null) => {
      const locked = applyLock(ratio);
      if (locked !== null) {
        commitBbox(resizeBboxAroundCentre(bbox, getSizeForRatio(locked)));
      }
    },
    [applyLock, bbox, commitBbox, getSizeForRatio]
  );

  // Locking captures the frame's current ratio (a built-in's exact value when it matches one).
  const onLockToggle = useCallback(
    () => applyLock(options.aspectLocked ? null : bboxRatio),
    [applyLock, bboxRatio, options.aspectLocked]
  );

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
          <Flex flex="1" minW="7.5rem">
            <AspectRatioPicker
              currentSize={currentSize}
              getSizeForRatio={getSizeForRatio}
              lockedRatio={lockedRatio}
              presets={aspectRatioPresets}
              onChange={onAspectRatioChange}
            />
          </Flex>
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
    <ApplyCancelBar applyDisabled={disabled} cancelDisabled={disabled} onApply={onApply} onCancel={onCancel}>
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
    POSITION_GROUP,
    { body: TransformScaleSettings, id: 'transform-scale', labelKey: 'widgets.transform.scale' },
  ],
  id: 'transform',
};
