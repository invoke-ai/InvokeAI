import { createExternalStore } from '@platform/state/externalStore';

export interface ModelLoadInfo {
  label: string;
}

const store = createExternalStore<{ activeLoads: ModelLoadInfo[] }>({ activeLoads: [] });

const getModelLoadLabel = (payload: unknown): string => {
  const record = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {};
  const config = record.config && typeof record.config === 'object' ? (record.config as Record<string, unknown>) : {};
  const name = typeof config.name === 'string' && config.name.trim() ? config.name.trim() : 'model';
  const extras = [config.base, config.type, record.submodel_type].filter(
    (value): value is string => typeof value === 'string' && value.trim().length > 0
  );

  return extras.length ? `${name} (${extras.join(', ')})` : name;
};

export interface ModelLoadActivitySink {
  started(payload: unknown): void;
  completed(payload: unknown): void;
  reset(): void;
}

export const modelLoadActivitySink: ModelLoadActivitySink = {
  started(payload: unknown): void {
    store.patchSnapshot({ activeLoads: [...store.getSnapshot().activeLoads, { label: getModelLoadLabel(payload) }] });
  },
  completed(payload: unknown): void {
    const label = getModelLoadLabel(payload);
    const { activeLoads } = store.getSnapshot();
    const index = activeLoads.findIndex((load) => load.label === label);

    store.patchSnapshot({
      activeLoads: index >= 0 ? activeLoads.filter((_, loadIndex) => loadIndex !== index) : activeLoads.slice(1),
    });
  },
  reset(): void {
    store.patchSnapshot({ activeLoads: [] });
  },
};

export const useModelLoads = (): ModelLoadInfo[] => store.useSelector((snapshot) => snapshot.activeLoads);

export const getModelLoads = (): ModelLoadInfo[] => store.getSnapshot().activeLoads;
