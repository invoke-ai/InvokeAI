import type { ReactNode } from 'react';

import { createContext, use } from 'react';

/** App supplies hint preferences/actions because Platform cannot import Workbench. */
export interface FeatureHintsAdapter {
  enabled: boolean;
  /** Turns hints off from inside a card; null when the host cannot persist preferences. */
  onDisable: (() => void) | null;
}

/** Default to disabled without a provider so isolated trees remain unchanged. */
const DEFAULT_FEATURE_HINTS_ADAPTER: FeatureHintsAdapter = {
  enabled: false,
  onDisable: null,
};

const FeatureHintsContext = createContext<FeatureHintsAdapter>(DEFAULT_FEATURE_HINTS_ADAPTER);

export const FeatureHintsProvider = ({ adapter, children }: { adapter: FeatureHintsAdapter; children: ReactNode }) => (
  <FeatureHintsContext value={adapter}>{children}</FeatureHintsContext>
);

export const useFeatureHints = (): FeatureHintsAdapter => use(FeatureHintsContext);
