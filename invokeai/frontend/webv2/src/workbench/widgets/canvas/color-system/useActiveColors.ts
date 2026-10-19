/** Share one persisted color pair across pane, properties, hotkeys, and eyedropper through these hooks. */

import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

import type { ActiveColorPair, ActiveColorTarget } from './colorPair';

import {
  areColorPairsEqual,
  areColorPalettesEqual,
  CANVAS_ACTIVE_COLOR_TARGET_KEY,
  CANVAS_ACTIVE_COLORS_KEY,
  CANVAS_COLOR_PALETTE_KEY,
  DEFAULT_COLOR_PAIR,
  readActiveColorPair,
  readActiveColorTarget,
  readColorPalette,
  swapColorPair,
  withoutPaletteColor,
  withPairColor,
  withPaletteColor,
} from './colorPair';

export const useActiveColorPair = (): ActiveColorPair =>
  useActiveProjectSelector(
    (project) => readActiveColorPair(getProjectWidgetValues(project, 'canvas')),
    areColorPairsEqual
  );

export const useActiveColorTarget = (): ActiveColorTarget =>
  useActiveProjectSelector((project) => readActiveColorTarget(getProjectWidgetValues(project, 'canvas')), Object.is);

export const useColorPalette = (): string[] =>
  useActiveProjectSelector(
    (project) => readColorPalette(getProjectWidgetValues(project, 'canvas')),
    areColorPalettesEqual
  );

export interface ActiveColorCommands {
  addPaletteColor: (color: string) => void;
  /** Writes an eyedropper result to whichever target is active. */
  applySampledColor: (color: string) => void;
  removePaletteColor: (color: string) => void;
  resetPair: () => void;
  setPairColor: (target: ActiveColorTarget, color: string) => void;
  setTarget: (target: ActiveColorTarget) => void;
  swapPair: () => void;
}

export const useActiveColorCommands = (): ActiveColorCommands => {
  const { widgets } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();

  return useMemo(() => {
    const values = (): Record<string, unknown> => getProjectWidgetValues(queries.getSnapshot().activeProject, 'canvas');
    // A no-op edit must not dispatch: the eyedropper re-samples per pointermove,
    // and an unchanged patch would still mint a new project snapshot.
    const patchPair = (current: ActiveColorPair, next: ActiveColorPair): void => {
      if (!areColorPairsEqual(current, next)) {
        widgets.patchValues('canvas', { [CANVAS_ACTIVE_COLORS_KEY]: next });
      }
    };
    const patchPalette = (current: readonly string[], next: string[]): void => {
      if (!areColorPalettesEqual(current, next)) {
        widgets.patchValues('canvas', { [CANVAS_COLOR_PALETTE_KEY]: next });
      }
    };

    return {
      addPaletteColor: (color) => {
        const palette = readColorPalette(values());
        patchPalette(palette, withPaletteColor(palette, color));
      },
      applySampledColor: (color) => {
        const current = values();
        const pair = readActiveColorPair(current);
        patchPair(pair, withPairColor(pair, readActiveColorTarget(current), color));
      },
      removePaletteColor: (color) => {
        const palette = readColorPalette(values());
        patchPalette(palette, withoutPaletteColor(palette, color));
      },
      resetPair: () => patchPair(readActiveColorPair(values()), { ...DEFAULT_COLOR_PAIR }),
      setPairColor: (target, color) => {
        const pair = readActiveColorPair(values());
        patchPair(pair, withPairColor(pair, target, color));
      },
      setTarget: (target) => {
        if (readActiveColorTarget(values()) !== target) {
          widgets.patchValues('canvas', { [CANVAS_ACTIVE_COLOR_TARGET_KEY]: target });
        }
      },
      swapPair: () => {
        const pair = readActiveColorPair(values());
        patchPair(pair, swapColorPair(pair));
      },
    };
  }, [queries, widgets]);
};
