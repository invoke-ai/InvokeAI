import type { CanvasLayerContract, LayerTransform, TransformAnchor, Vec2 } from '@workbench/canvas-engine/api';
import type { ToolFormProps } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { Box, Flex, Stack } from '@chakra-ui/react';
import { AspectRatioLockButton } from '@features/generation/components';
import { IconButton } from '@platform/ui/Button';
import {
  flipTransform,
  hittableLayerRect,
  layerPointToDocument,
  localHandlePoint,
  lookupDocumentLeaf,
  moveLocalPointTo,
  retransformAbout,
  scaleAxis,
  scaleForLength,
  transformedSize,
} from '@workbench/canvas-engine/api';
import {
  useCanvasActiveTool,
  useCanvasDocumentEditingLocked,
  useCanvasHasFloatingSelection,
  useTransformOptions,
  useTransformSession,
} from '@workbench/widgets/canvas/engineStoreHooks';
import { FormNumberField, useNumberCommit } from '@workbench/widgets/canvas/tool-presentation/FormControls';
import { PropertyControlRow, PropertyRowLabel } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { TrianglesCenterlineDashedVerticalIcon, TrianglesCenterlineDashedHorizontalIcon } from 'lucide-react';
import { type ReactNode, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { ReferencePointPicker } from './ReferencePointPicker';

const RAD_TO_DEG = 180 / Math.PI;
const DEG_TO_RAD = Math.PI / 180;
const LABEL_WIDTH = '6rem';
const ORIGIN: Vec2 = { x: 0, y: 0 };
const MIN_SIZE_PX = 1;

const round2 = (n: number): number => Math.round(n * 100) / 100;
/** Degrees in (-180, 180], so 270 reads and stores as -90. */
const wrapDegrees = (degrees: number): number => {
  const wrapped = ((((degrees + 180) % 360) + 360) % 360) - 180;
  return wrapped === -180 ? 180 : wrapped;
};

interface SelectedLeaf {
  layer: CanvasLayerContract;
  locked: boolean;
  /** Legacy gradients without an extent are document-sized, so the content box reads the document size. */
  documentWidth: number;
  documentHeight: number;
}

type Axis = 'x' | 'y';

/** The shared layer-transform write path: a live session absorbs edits; otherwise each field is one patch. */
export const useLayerTransformEditor = (engine: ToolFormProps['engine'], labelKey = 'widgets.transform.edit') => {
  const { t } = useTranslation();
  const commitPrepared = usePreparedCommit(engine);
  const session = useTransformSession(engine);
  const options = useTransformOptions(engine);
  const activeTool = useCanvasActiveTool(engine);
  const hasFloat = useCanvasHasFloatingSelection(engine);
  const locked = useCanvasDocumentEditingLocked(engine);
  const selected = useActiveProjectSelector(
    (project): SelectedLeaf | null => {
      const { document } = project.canvas;
      const leaf = document.selectedLayerId ? lookupDocumentLeaf(document, document.selectedLayerId) : null;
      return leaf
        ? {
            documentHeight: document.height,
            documentWidth: document.width,
            layer: leaf.layer,
            locked: leaf.effectiveLocked,
          }
        : null;
    },
    (a, b) =>
      a?.layer === b?.layer &&
      a?.locked === b?.locked &&
      a?.documentWidth === b?.documentWidth &&
      a?.documentHeight === b?.documentHeight
  );
  const layer = selected?.layer ?? null;
  const documentWidth = selected?.documentWidth ?? 0;
  const documentHeight = selected?.documentHeight ?? 0;
  // Walks a vector layer's paths, so it runs only when the layer contract or document size changes.
  const contentRect = useMemo(
    () => (layer ? hittableLayerRect(layer, { height: documentHeight, width: documentWidth }) : null),
    [documentHeight, documentWidth, layer]
  );
  const transform = session?.transform ?? layer?.transform ?? null;
  const rect = layer && (!session || session.layerId === layer.id) ? contentRect : null;
  const editable =
    !locked &&
    !session?.processing &&
    selected !== null &&
    !selected.locked &&
    (activeTool !== 'transform' || hasFloat || session?.layerId === selected.layer.id);
  const local = rect ? localHandlePoint(rect, options.referencePoint) : ORIGIN;
  const position = transform ? layerPointToDocument(transform, local) : null;
  const size = transform && rect ? transformedSize(transform, rect) : null;

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
        model.prepare({ id: selected.layer.id, patch: { transform: next }, type: 'patch' })
      );
    },
    [commitPrepared, editable, engine, labelKey, selected, session, t]
  );

  // Moves keep the origin on whole pixels; the readout shows where the reference point actually lands.
  const setPosition = useCallback(
    (axis: Axis, value: number) => {
      if (!transform || !position) {
        return;
      }
      const moved = moveLocalPointTo(transform, local, { ...position, [axis]: value });
      patch({ [axis]: Math.round(moved[axis]) });
    },
    [local, patch, position, transform]
  );
  const setSize = useCallback(
    (axis: Axis, length: number) => {
      if (!transform || !rect) {
        return;
      }
      const own =
        axis === 'x' ? { rect: rect.width, scale: transform.scaleX } : { rect: rect.height, scale: transform.scaleY };
      const scale = scaleForLength(Math.max(MIN_SIZE_PX, length), own.rect, own.scale);
      patch(retransformAbout(transform, local, scaleAxis(transform, axis, scale, options.aspectLinked)));
    },
    [local, options.aspectLinked, patch, rect, transform]
  );
  const flip = useCallback(
    (direction: 'horizontal' | 'vertical') => {
      if (transform && rect) {
        patch(flipTransform(transform, rect, direction));
      }
    },
    [patch, rect, transform]
  );
  const setRotation = useCallback(
    (degrees: number) => {
      if (transform) {
        patch(retransformAbout(transform, local, { rotation: wrapDegrees(degrees) * DEG_TO_RAD }));
      }
    },
    [local, patch, transform]
  );
  const setReferencePoint = useCallback(
    (referencePoint: TransformAnchor) => engine.interaction.set('transformOptions', { ...options, referencePoint }),
    [engine, options]
  );
  const toggleAspectLink = useCallback(
    () => engine.interaction.set('transformOptions', { ...options, aspectLinked: !options.aspectLinked }),
    [engine, options]
  );
  return {
    aspectLinked: options.aspectLinked,
    editable,
    flip,
    hasContentBox: rect !== null,
    position,
    referencePoint: options.referencePoint,
    setPosition,
    setReferencePoint,
    setRotation,
    setSize,
    size,
    toggleAspectLink,
    transform,
  };
};

type LayerTransformEditor = ReturnType<typeof useLayerTransformEditor>;

const LabelWithControl = ({
  children,
  label,
  shown = true,
}: {
  children: ReactNode;
  label: string;
  shown?: boolean;
}) => (
  <Flex align="center" display={shown ? 'flex' : 'none'} flexShrink={0} gap="1" w={LABEL_WIDTH}>
    <Box flex="1" minW="0">
      <PropertyRowLabel label={label} />
    </Box>
    {children}
  </Flex>
);

export const PositionFields = ({
  disabled,
  referencePoint,
  showLabel,
  x,
  y,
  onReferencePointChange,
  onX,
  onY,
}: {
  disabled: boolean;
  referencePoint?: TransformAnchor;
  showLabel: boolean;
  x: string;
  y: string;
  onReferencePointChange?: (anchor: TransformAnchor) => void;
  onX: (value: number) => void;
  onY: (value: number) => void;
}) => {
  const { t } = useTranslation();
  const onXCommit = useNumberCommit(onX);
  const onYCommit = useNumberCommit(onY);
  const picker =
    referencePoint && onReferencePointChange ? (
      <ReferencePointPicker value={referencePoint} onChange={onReferencePointChange} />
    ) : null;
  return (
    <Flex align="center" gap="2" w="full">
      <LabelWithControl label={t('widgets.transform.position')} shown={showLabel}>
        {showLabel ? picker : null}
      </LabelWithControl>
      <FormNumberField
        aria-label={t('widgets.canvas.toolOptions.positionX')}
        disabled={disabled}
        label={t('widgets.canvas.toolOptions.positionX')}
        value={x}
        onValueCommit={onXCommit}
      />
      <FormNumberField
        aria-label={t('widgets.canvas.toolOptions.positionY')}
        disabled={disabled}
        label={t('widgets.canvas.toolOptions.positionY')}
        value={y}
        onValueCommit={onYCommit}
      />
      {showLabel ? null : picker}
    </Flex>
  );
};

export const layerPositionText = (editor: LayerTransformEditor, axis: Axis): string => {
  if (!editor.position || !editor.transform) {
    return '';
  }
  return String(editor.hasContentBox ? round2(editor.position[axis]) : Math.round(editor.transform[axis]));
};

export const LayerPositionRow = ({ engine, showLabel }: ToolFormProps & { showLabel: boolean }) => {
  const editor = useLayerTransformEditor(engine);
  const onX = useCallback((value: number) => editor.setPosition('x', value), [editor]);
  const onY = useCallback((value: number) => editor.setPosition('y', value), [editor]);
  return (
    <PositionFields
      disabled={editor.transform === null || !editor.editable}
      showLabel={showLabel}
      referencePoint={editor.referencePoint}
      x={layerPositionText(editor, 'x')}
      y={layerPositionText(editor, 'y')}
      onReferencePointChange={editor.setReferencePoint}
      onX={onX}
      onY={onY}
    />
  );
};

export const LayerTransformRows = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const editor = useLayerTransformEditor(engine);
  const { editable, flip, setRotation, setSize, size, transform } = editor;
  const onWidth = useNumberCommit(useCallback((value: number) => setSize('x', value), [setSize]));
  const onHeight = useNumberCommit(useCallback((value: number) => setSize('y', value), [setSize]));
  const onRotation = useNumberCommit(setRotation);
  const onFlipHorizontal = useCallback(() => flip('horizontal'), [flip]);
  const onFlipVertical = useCallback(() => flip('vertical'), [flip]);
  const flipDisabled = !editable || !editor.hasContentBox;
  return (
    <Stack gap="2">
      <Flex align="center" gap="2" w="full">
        <LabelWithControl label={t('widgets.properties.rows.size')}>
          <AspectRatioLockButton isLocked={editor.aspectLinked} onToggle={editor.toggleAspectLink} />
        </LabelWithControl>
        <FormNumberField
          aria-label={t('widgets.canvas.toolOptions.frameWidth')}
          disabled={!editable || !size}
          label={t('widgets.canvas.toolOptions.frameWidth')}
          min={MIN_SIZE_PX}
          relative
          value={size ? String(round2(size.width)) : ''}
          onValueCommit={onWidth}
        />
        <FormNumberField
          aria-label={t('widgets.canvas.toolOptions.frameHeight')}
          disabled={!editable || !size}
          label={t('widgets.canvas.toolOptions.frameHeight')}
          min={MIN_SIZE_PX}
          relative
          value={size ? String(round2(size.height)) : ''}
          onValueCommit={onHeight}
        />
      </Flex>
      <PropertyControlRow label={t('widgets.transform.rotation')} labelWidth={LABEL_WIDTH}>
        <Flex align="center" gap="1" gridColumn="2 / -1">
          <FormNumberField
            aria-label={t('widgets.canvas.toolOptions.rotation')}
            disabled={!editable}
            suffix="°"
            value={transform ? String(round2(wrapDegrees(transform.rotation * RAD_TO_DEG))) : ''}
            onValueCommit={onRotation}
          />
          <IconButton
            aria-label={t('widgets.transform.flipHorizontal')}
            disabled={flipDisabled}
            ms="1"
            tooltip={t('widgets.transform.flipHorizontal')}
            variant="ghost"
            onClick={onFlipHorizontal}
          >
            <TrianglesCenterlineDashedVerticalIcon />
          </IconButton>
          <IconButton
            aria-label={t('widgets.transform.flipVertical')}
            disabled={flipDisabled}
            tooltip={t('widgets.transform.flipVertical')}
            variant="ghost"
            onClick={onFlipVertical}
          >
            <TrianglesCenterlineDashedHorizontalIcon />
          </IconButton>
        </Flex>
      </PropertyControlRow>
    </Stack>
  );
};
