import { useSyncExternalStore } from 'react';

export interface ModelLoadInfo {
  label: string;
}

const listeners = new Set<() => void>();

let activeLoads: ModelLoadInfo[] = [];

const emit = (): void => {
  for (const listener of listeners) {
    listener();
  }
};

const subscribe = (listener: () => void): (() => void) => {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
};

const getModelLoadLabel = (payload: unknown): string => {
  const record = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {};
  const config = record.config && typeof record.config === 'object' ? (record.config as Record<string, unknown>) : {};
  const name = typeof config.name === 'string' && config.name.trim() ? config.name.trim() : 'model';
  const extras = [config.base, config.type, record.submodel_type].filter(
    (value): value is string => typeof value === 'string' && value.trim().length > 0
  );

  return extras.length ? `${name} (${extras.join(', ')})` : name;
};

export const modelLoadStore = {
  started(payload: unknown): void {
    activeLoads = [...activeLoads, { label: getModelLoadLabel(payload) }];
    emit();
  },
  completed(payload: unknown): void {
    const label = getModelLoadLabel(payload);
    const index = activeLoads.findIndex((load) => load.label === label);

    if (index >= 0) {
      activeLoads = activeLoads.filter((_, loadIndex) => loadIndex !== index);
    } else {
      activeLoads = activeLoads.slice(1);
    }

    emit();
  },
  reset(): void {
    if (activeLoads.length > 0) {
      activeLoads = [];
      emit();
    }
  },
};

export const useModelLoads = (): ModelLoadInfo[] => useSyncExternalStore(subscribe, () => activeLoads);
