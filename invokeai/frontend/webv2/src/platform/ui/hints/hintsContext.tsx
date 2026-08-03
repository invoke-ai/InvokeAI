import type { ReactNode } from 'react';

import { createContext, use } from 'react';

/**
 * Feature hints' host port. Platform may not import Workbench, so the enabled
 * preference and the "turn these off" command are pushed in from App — the same
 * direction `I18nController` and `ThemeController` feed their platform
 * singletons. Not a test seam; no second adapter is expected.
 */
export interface FeatureHintsAdapter {
  enabled: boolean;
  /** Turns hints off from inside a card; null when the host cannot persist preferences. */
  onDisable: (() => void) | null;
}

/**
 * Inert by default: a tree with no provider renders exactly as it did before
 * hints existed, so unit tests and isolated browser tests need no setup.
 */
const DEFAULT_FEATURE_HINTS_ADAPTER: FeatureHintsAdapter = {
  enabled: false,
  onDisable: null,
};

const FeatureHintsContext = createContext<FeatureHintsAdapter>(DEFAULT_FEATURE_HINTS_ADAPTER);

export const FeatureHintsProvider = ({ adapter, children }: { adapter: FeatureHintsAdapter; children: ReactNode }) => (
  <FeatureHintsContext value={adapter}>{children}</FeatureHintsContext>
);

export const useFeatureHints = (): FeatureHintsAdapter => use(FeatureHintsContext);
