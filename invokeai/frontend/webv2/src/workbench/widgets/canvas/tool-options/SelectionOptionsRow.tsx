import type { SelectionOp } from '@workbench/canvas-engine/api';
import type { ToolFormProps } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { HStack } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { isLeafPixelEditEligible, lookupDocumentLeaf } from '@workbench/canvas-engine/api';
import { useNotify } from '@workbench/useNotify';
import { useCanvasHasSelection } from '@workbench/widgets/canvas/engineStoreHooks';
import { reportLayerOperation } from '@workbench/widgets/canvas/useStructuralCommit';
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

/**
 * Commands over the live selection. Select all always works; fill, erase and
 * lift need an eligible (unlocked, visible) paint layer — the same rule the
 * engine enforces; invert and deselect need only a selection.
 */
export const SelectionActions = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const hasSelection = useCanvasHasSelection(engine);
  const canPaintTarget = useActiveProjectSelector((project) => {
    const { document } = project.canvas;
    return isLeafPixelEditEligible(lookupDocumentLeaf(document, document.selectedLayerId ?? ''));
  });
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
  const needsPaintTarget =
    needsSelection ?? (canPaintTarget ? null : t('widgets.canvas.toolOptions.selectionNeedsPaintLayer'));
  return (
    <HStack flexWrap="wrap" gap="1">
      <SelectionAction disabledReason={null} label={t('widgets.canvas.toolOptions.selectAll')} onClick={onSelectAll} />
      <SelectionAction
        disabledReason={needsPaintTarget}
        label={t('widgets.canvas.toolOptions.fillSelection')}
        onClick={onFill}
      />
      <SelectionAction
        disabledReason={needsPaintTarget}
        label={t('widgets.canvas.toolOptions.eraseSelection')}
        onClick={onErase}
      />
      <SelectionAction
        disabledReason={needsPaintTarget}
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
