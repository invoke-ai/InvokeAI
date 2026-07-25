import type { SelectionOp } from '@workbench/canvas-engine/api';

import { HStack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui';
import { isLayerPixelEditEligible } from '@workbench/canvas-engine/api';
import { useCanvasHasSelection } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

const OP_MODES: readonly SelectionOp[] = ['replace', 'add', 'subtract', 'intersect'];

const OP_MODE_LABEL_KEYS: Record<SelectionOp, string> = {
  add: 'widgets.canvas.toolOptions.selectionAdd',
  intersect: 'widgets.canvas.toolOptions.selectionIntersect',
  replace: 'widgets.canvas.toolOptions.selectionReplace',
  subtract: 'widgets.canvas.toolOptions.selectionSubtract',
};

interface OpModeButtonProps {
  mode: SelectionOp;
  active: boolean;
  onSelect: (mode: SelectionOp) => void;
}

/** One op-mode button with a stable click handler (avoids a per-render closure in the map). */
const OpModeButton = ({ active, mode, onSelect }: OpModeButtonProps) => {
  const { t } = useTranslation();
  const onClick = useCallback(() => onSelect(mode), [mode, onSelect]);
  return (
    <Button aria-pressed={active} size="xs" variant={active ? 'solid' : 'ghost'} onClick={onClick}>
      {t(OP_MODE_LABEL_KEYS[mode])}
    </Button>
  );
};

export interface SelectionOptionsRowProps extends ToolOptionsComponentProps {
  /** The tool's persistent boolean op mode. */
  mode: SelectionOp;
  /** Writes the tool's op mode back to its own options store. */
  onModeChange: (mode: SelectionOp) => void;
  /** Shown when there is no live selection — tells the user what the gesture does. */
  hintKey: string;
}

/**
 * The controls every pixel-selection tool shares: the boolean op-mode selector
 * (replace / add / subtract / intersect — also settable transiently by holding
 * shift / alt / shift+alt while committing) plus fill / erase / invert /
 * deselect actions.
 *
 * Fill and erase require an eligible (unlocked, visible) paint layer selected;
 * invert and deselect only require a live selection. Reads and writes the
 * engine's transient selection state directly — no reducer mirror. The op mode
 * itself lives in each tool's own options store, so it is passed in rather than
 * read here.
 */
export const SelectionOptionsRow = ({ engine, hintKey, mode, onModeChange }: SelectionOptionsRowProps) => {
  const { t } = useTranslation();
  const hasSelection = useCanvasHasSelection(engine);

  // Whether the selected layer can receive a masked fill/erase (paint, unlocked,
  // visible). Same eligibility the engine enforces; used to disable the buttons.
  const canPaintTarget = useActiveProjectSelector((project) => {
    const { document } = project.canvas;
    const layer = document.selectedLayerId
      ? document.layers.find((entry) => entry.id === document.selectedLayerId)
      : undefined;
    return isLayerPixelEditEligible(layer);
  });

  const onFill = useCallback(() => engine.selection.fillSelection(), [engine]);
  const onErase = useCallback(() => engine.selection.eraseSelection(), [engine]);
  const onInvert = useCallback(() => engine.selection.invertSelection(), [engine]);
  const onDeselect = useCallback(() => engine.selection.deselect(), [engine]);
  const onLiftToLayer = useCallback(() => engine.selection.liftSelectionToLayer(), [engine]);

  const canEdit = hasSelection && canPaintTarget;

  return (
    <HStack align="center" gap="3">
      <HStack align="center" gap="1" role="group" aria-label={t('widgets.canvas.toolOptions.selectionMode')}>
        {OP_MODES.map((opMode) => (
          <OpModeButton key={opMode} active={mode === opMode} mode={opMode} onSelect={onModeChange} />
        ))}
      </HStack>
      <HStack align="center" gap="1">
        <Button disabled={!canEdit} size="xs" variant="ghost" onClick={onFill}>
          {t('widgets.canvas.toolOptions.fillSelection')}
        </Button>
        <Button disabled={!canEdit} size="xs" variant="ghost" onClick={onErase}>
          {t('widgets.canvas.toolOptions.eraseSelection')}
        </Button>
        <Button disabled={!canEdit} size="xs" variant="ghost" onClick={onLiftToLayer}>
          {t('widgets.canvas.toolOptions.liftSelectionToLayer')}
        </Button>
        <Button disabled={!hasSelection} size="xs" variant="ghost" onClick={onInvert}>
          {t('widgets.canvas.toolOptions.invertSelection')}
        </Button>
        <Button disabled={!hasSelection} size="xs" variant="ghost" onClick={onDeselect}>
          {t('widgets.canvas.toolOptions.deselect')}
        </Button>
      </HStack>
      {!hasSelection ? (
        <Text color="fg.muted" fontSize="2xs">
          {t(hintKey)}
        </Text>
      ) : null}
    </HStack>
  );
};
