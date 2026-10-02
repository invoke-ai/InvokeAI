/**
 * Feed backend model grid policy into the model-agnostic engine. {@link resolveModelGrid} distinguishes
 * unavailable policy so persistent writers wait while readers may use defaults; {@link useModelGridSize} updates
 * readers when policy arrives.
 */

import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { getDimensionGrid } from '@features/generation/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { useCallback } from 'react';

/** Default grid when no model is selected. */
export const DEFAULT_MODEL_GRID = 8;

/**
 * The grid the selected architecture declares, `DEFAULT_MODEL_GRID` when no model is selected, or
 * `null` while the backend has not answered for that architecture.
 */
export const resolveModelGrid = (base: string | null | undefined, variant?: unknown): number | null =>
  base ? getDimensionGrid(base, variant) : DEFAULT_MODEL_GRID;

export const gridSizeForModelBase = (base: string | null | undefined, variant?: unknown): number =>
  resolveModelGrid(base, variant) ?? DEFAULT_MODEL_GRID;

/**
 * Resolve grid inside the capability selector so table updates recompute it even when the persisted base stays
 * unchanged and compiler memoization would retain a fallback.
 */
export const useModelGridSize = (base: string | null | undefined, variant?: string | null): number =>
  useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(() => gridSizeForModelBase(base, variant), [base, variant])
  );
