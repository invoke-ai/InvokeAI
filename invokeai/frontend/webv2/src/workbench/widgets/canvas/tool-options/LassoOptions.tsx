import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { SelectionOp } from '@workbench/canvas-engine/types';

import { HStack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { useCanvasHasSelection, useLassoOptions } from '@workbench/widgets/canvas/engineStoreHooks';
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

/** One op-mode button with a stable click handler (avoids a per-render closure in the map). */
const OpModeButton = ({ engine, mode, active }: { engine: CanvasEngine; mode: SelectionOp; active: boolean }) => {
  const { t } = useTranslation();
  const onClick = useCallback(() => engine.stores.lassoOptions.set({ mode }), [engine, mode]);
  return (
    <Button aria-pressed={active} size="xs" variant={active ? 'solid' : 'ghost'} onClick={onClick}>
      {t(OP_MODE_LABEL_KEYS[mode])}
    </Button>
  );
};

/**
 * Lasso tool options: the boolean op-mode selector (replace / add / subtract /
 * intersect — also settable transiently by holding shift / alt / shift+alt while
 * committing a path) plus fill / erase / invert / deselect actions. Fill and
 * erase require an eligible (unlocked, visible) paint layer selected; invert and
 * deselect only require a live selection. Reads/writes the engine's transient
 * selection state directly — no reducer mirror.
 */
export const LassoOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useLassoOptions(engine);
  const hasSelection = useCanvasHasSelection(engine);

  // Whether the selected layer can receive a masked fill/erase (paint, unlocked,
  // visible). Same eligibility the engine enforces; used to disable the buttons.
  const canPaintTarget = useActiveProjectSelector((project) => {
    const { document } = project.canvas;
    const layer = document.selectedLayerId
      ? document.layers.find((entry) => entry.id === document.selectedLayerId)
      : undefined;
    return !!layer && layer.type === 'raster' && layer.source.type === 'paint' && !layer.isLocked && layer.isEnabled;
  });

  const onFill = useCallback(() => engine.fillSelection(), [engine]);
  const onErase = useCallback(() => engine.eraseSelection(), [engine]);
  const onInvert = useCallback(() => engine.invertSelection(), [engine]);
  const onDeselect = useCallback(() => engine.deselect(), [engine]);

  const canEdit = hasSelection && canPaintTarget;

  return (
    <HStack align="center" gap="3">
      <HStack align="center" gap="1" role="group" aria-label={t('widgets.canvas.toolOptions.selectionMode')}>
        {OP_MODES.map((mode) => (
          <OpModeButton key={mode} active={options.mode === mode} engine={engine} mode={mode} />
        ))}
      </HStack>
      <HStack align="center" gap="1">
        <Button disabled={!canEdit} size="xs" variant="ghost" onClick={onFill}>
          {t('widgets.canvas.toolOptions.fillSelection')}
        </Button>
        <Button disabled={!canEdit} size="xs" variant="ghost" onClick={onErase}>
          {t('widgets.canvas.toolOptions.eraseSelection')}
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
          {t('widgets.canvas.toolOptions.lassoHint')}
        </Text>
      ) : null}
    </HStack>
  );
};
