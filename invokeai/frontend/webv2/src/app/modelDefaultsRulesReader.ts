import type { ArchitectureCapabilitiesSnapshot } from '@features/generation/runtime';
import type { DefaultComponentSlotReader, SchedulerPolicyReader, ModelDefaultsRulesState } from '@features/models';

import {
  ensureArchitectureCapabilitiesLoaded,
  getArchitectureCapabilitiesSnapshot,
  subscribeArchitectureCapabilities,
} from '@features/generation/runtime';
import {
  getDefaultComponentSlotPolicies,
  getDefaultGenerateSettings,
  isVaeCompatibleWithGenerateModel,
  getGenerationUiPolicy,
  getSchedulerOptions,
} from '@features/generation/settings';

type GenerateMainModel = Parameters<typeof getDefaultGenerateSettings>[0] & { type: 'main' };

/** Generate's per-base rules; the stored default is dropped so the base's own default is read. */
const readSchedulerPolicy: SchedulerPolicyReader = (model, currentValue) => {
  const mainModel = { ...model, default_settings: null, type: 'main' } as GenerateMainModel;

  return {
    appliesToGraph: getGenerationUiPolicy(mainModel).schedulerVisible,
    defaultValue: getDefaultGenerateSettings(mainModel).scheduler,
    options: getSchedulerOptions(mainModel, currentValue ?? undefined),
  };
};

const readDefaultComponentSlots: DefaultComponentSlotReader = (model, components, catalog) =>
  getDefaultComponentSlotPolicies(
    { ...model, default_settings: { components }, type: 'main' } as GenerateMainModel,
    catalog
  );

const toPolicyState = (snapshot: ArchitectureCapabilitiesSnapshot): ModelDefaultsRulesState => {
  if (snapshot.status === 'loaded' && snapshot.revision > 0) {
    return {
      acceptsDefaultVae: (model, candidate) =>
        isVaeCompatibleWithGenerateModel({ ...model, type: 'main' } as GenerateMainModel, candidate as never),
      read: (model, currentValue) => readSchedulerPolicy(model, currentValue),
      readComponentSlots: (model, components, catalog) => readDefaultComponentSlots(model, components, catalog),
      status: 'ready',
    };
  }
  return snapshot.status === 'error' ? { status: 'unavailable' } : { status: 'loading' };
};

/**
 * Publishes the rules' state now and whenever the architecture table's status or revision changes, with a new reader
 * per revision so memoized callers recompute. Lives for the page, like the table's own store.
 */
export const startModelDefaultsRulesReader = (publish: (state: ModelDefaultsRulesState) => void): void => {
  let published: Pick<ArchitectureCapabilitiesSnapshot, 'revision' | 'status'> | null = null;
  const sync = () => {
    const snapshot = getArchitectureCapabilitiesSnapshot();
    if (published?.revision === snapshot.revision && published.status === snapshot.status) {
      return;
    }
    published = { revision: snapshot.revision, status: snapshot.status };
    publish(toPolicyState(snapshot));
  };
  sync();
  subscribeArchitectureCapabilities(sync);
};

/** Joins or starts the table's load; it never rejects (see `ensureArchitectureCapabilitiesLoaded`). */
export const ensureDefaultsRulesLoaded = (): Promise<void> => ensureArchitectureCapabilitiesLoaded();
