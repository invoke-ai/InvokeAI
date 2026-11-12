import type { CanvasDocumentContractV3, SelectionOp, SelectionPixelEditRefusal } from '@workbench/canvas-engine/api';
import type { ToolFormProps } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';
import type { TFunction } from 'i18next';

import { HStack } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  getDocumentNode,
  isLeafPixelEditEligible,
  lookupDocumentLeaf,
  selectionPixelEditRefusal,
} from '@workbench/canvas-engine/api';
import { useNotify } from '@workbench/useNotify';
import { useCanvasHasSelection, useTransformSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { describeLayerEditRefusal, reportLayerOperation } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { SquareIcon, SquareMinusIcon, SquarePlusIcon, SquaresIntersectIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const OP_MODES: readonly SelectionOp[] = ['replace', 'add', 'subtract', 'intersect'];

const OP_MODE_LABEL_KEYS: Record<SelectionOp, string> = {
  add: 'widgets.canvas.toolOptions.selectionAdd',
  intersect: 'widgets.canvas.toolOptions.selectionIntersect',
  replace: 'widgets.canvas.toolOptions.selectionReplace',
  subtract: 'widgets.canvas.toolOptions.selectionSubtract',
};

const OP_MODE_ICONS: Record<SelectionOp, typeof SquareIcon> = {
  add: SquarePlusIcon,
  intersect: SquaresIntersectIcon,
  replace: SquareIcon,
  subtract: SquareMinusIcon,
};

const OpModeButton = ({
  active,
  mode,
  onSelect,
}: {
  active: boolean;
  mode: SelectionOp;
  onSelect: (mode: SelectionOp) => void;
}) => {
  const { t } = useTranslation();
  const onClick = useCallback(() => onSelect(mode), [mode, onSelect]);
  const label = t(OP_MODE_LABEL_KEYS[mode]);
  const Icon = OP_MODE_ICONS[mode];
  return (
    <Tooltip content={label}>
      <IconButton aria-label={label} aria-pressed={active} variant={active ? 'solid' : 'ghost'} onClick={onClick}>
        <Icon />
      </IconButton>
    </Tooltip>
  );
};

/**
 * Share selection operation mode and empty-selection hints; each tool stores its mode, with transient Shift/Alt
 * overrides.
 */
/** The four op-mode toggles alone; the form places them in a labelled row. */
export const SelectionOpModeButtons = ({
  mode,
  onModeChange,
}: {
  mode: SelectionOp;
  onModeChange: (mode: SelectionOp) => void;
}) => {
  const { t } = useTranslation();
  return (
    <HStack aria-label={t('widgets.canvas.toolOptions.selectionMode')} flexShrink={0} gap="1" role="group">
      {OP_MODES.map((opMode) => (
        <OpModeButton key={opMode} active={mode === opMode} mode={opMode} onSelect={onModeChange} />
      ))}
    </HStack>
  );
};

/**
 * A selection command; while unavailable, its tooltip says what is missing
 * instead of going quiet. The button is aria-disabled rather than natively
 * disabled so it still emits the pointer events the tooltip needs and stays
 * reachable by keyboard, where the reason reads as its description.
 */
const SelectionAction = ({
  disabledReason,
  label,
  onClick,
}: {
  disabledReason: string | null;
  label: string;
  onClick: () => void;
}) => {
  const guarded = useCallback(() => {
    if (disabledReason === null) {
      onClick();
    }
  }, [disabledReason, onClick]);
  return (
    <Tooltip content={disabledReason ?? ''} disabled={disabledReason === null}>
      <Button aria-disabled={disabledReason !== null} variant="ghost" onClick={guarded}>
        {label}
      </Button>
    </Tooltip>
  );
};

/** The selected leaf, `'group'` for a selected group, or null: what `selectionPixelEditRefusal` judges. */
const selectedLeafOrGroup = (document: CanvasDocumentContractV3) => {
  const node = getDocumentNode(document, document.selectedLayerId);
  return node ? (lookupDocumentLeaf(document, node.id) ?? 'group') : null;
};

/**
 * Why Fill or Erase is unavailable, in the words of the notice the engine would give: what is wrong and what to do.
 * `layer` names the layer that refuses.
 */
const describePixelRefusal = (
  status: Exclude<SelectionPixelEditRefusal, 'no-target' | 'unsupported'> | 'transform-session',
  layer: { id: string; name: string },
  t: TFunction
): string => {
  const { message, title } = describeLayerEditRefusal({ layerId: layer.id, layerName: layer.name, status }, t);
  return t('widgets.canvas.toolOptions.refusalReason', { message, title });
};

/**
 * Commands over the live selection. Select all always works; fill and erase
 * need a layer the brush and eraser would edit in place, refused the way they
 * are (the engine's own rule); lift needs an eligible paint or control layer;
 * invert and deselect need only a selection.
 */
export const SelectionActions = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const hasSelection = useCanvasHasSelection(engine);
  const canLiftTarget = useActiveProjectSelector((project) => {
    const { document } = project.canvas;
    return isLeafPixelEditEligible(lookupDocumentLeaf(document, document.selectedLayerId ?? ''));
  });
  const fillRefusal = useActiveProjectSelector((project) =>
    selectionPixelEditRefusal(selectedLeafOrGroup(project.canvas.document), 'fill')
  );
  const eraseRefusal = useActiveProjectSelector((project) =>
    selectionPixelEditRefusal(selectedLeafOrGroup(project.canvas.document), 'erase')
  );
  const selectedName = useActiveProjectSelector(
    (project) => getDocumentNode(project.canvas.document, project.canvas.document.selectedLayerId)?.name ?? ''
  );
  const sessionLayerId = useTransformSession(engine)?.layerId ?? null;
  const sessionName = useActiveProjectSelector((project) =>
    sessionLayerId === null ? '' : (getDocumentNode(project.canvas.document, sessionLayerId)?.name ?? '')
  );
  const onSelectAll = useCallback(() => engine.selection.selectAll(), [engine]);
  const onFill = useCallback(() => engine.selection.fillSelection(), [engine]);
  const onErase = useCallback(() => engine.selection.eraseSelection(), [engine]);
  const onInvert = useCallback(() => engine.selection.invertSelection(), [engine]);
  const onDeselect = useCallback(() => engine.selection.deselect(), [engine]);
  const onLiftToLayer = useCallback(() => {
    const result = engine.selection.liftSelectionToLayer();
    if (result.status !== 'created' && result.status !== 'empty') {
      reportLayerOperation(result.status, notify.error, t);
    }
  }, [engine, notify, t]);
  const needsSelection = hasSelection ? null : t('widgets.canvas.toolOptions.selectionNeedsSelection');
  // The engine's order: what the target refuses, then an open transform session.
  const pixelReason = (refusal: SelectionPixelEditRefusal | null): string | null =>
    needsSelection ??
    (refusal === 'no-target' || refusal === 'unsupported'
      ? t('widgets.canvas.toolOptions.selectionNeedsPixelLayer')
      : refusal
        ? describePixelRefusal(refusal, { id: '', name: selectedName }, t)
        : sessionLayerId !== null
          ? describePixelRefusal('transform-session', { id: sessionLayerId, name: sessionName }, t)
          : null);
  const fillReason = pixelReason(fillRefusal);
  const eraseReason = pixelReason(eraseRefusal);
  const liftReason =
    needsSelection ?? (canLiftTarget ? null : t('widgets.canvas.toolOptions.selectionNeedsPaintLayer'));
  return (
    <HStack flexWrap="wrap" gap="1">
      <SelectionAction disabledReason={null} label={t('widgets.canvas.toolOptions.selectAll')} onClick={onSelectAll} />
      <SelectionAction
        disabledReason={fillReason}
        label={t('widgets.canvas.toolOptions.fillSelection')}
        onClick={onFill}
      />
      <SelectionAction
        disabledReason={eraseReason}
        label={t('widgets.canvas.toolOptions.eraseSelection')}
        onClick={onErase}
      />
      <SelectionAction
        disabledReason={liftReason}
        label={t('widgets.canvas.toolOptions.liftSelectionToLayer')}
        onClick={onLiftToLayer}
      />
      <SelectionAction
        disabledReason={needsSelection}
        label={t('widgets.canvas.toolOptions.invertSelection')}
        onClick={onInvert}
      />
      <SelectionAction
        disabledReason={needsSelection}
        label={t('widgets.canvas.toolOptions.deselect')}
        onClick={onDeselect}
      />
    </HStack>
  );
};
