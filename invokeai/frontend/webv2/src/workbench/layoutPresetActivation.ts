import type { LayoutPreset, LayoutPresetId } from './layoutContracts';

import { loadWidget, preloadWidget } from './widgetRegistry';

interface LayoutPresetActivatorDependencies {
  apply: (presetId: LayoutPresetId) => void;
  getActiveProjectId: () => string;
  isCurrent: (preset: LayoutPreset) => boolean;
  load: (preset: LayoutPreset) => Promise<unknown>;
}

export const createLayoutPresetActivator = ({
  apply,
  getActiveProjectId,
  isCurrent,
  load,
}: LayoutPresetActivatorDependencies) => {
  let latestRequestId = 0;

  return {
    activate: async (preset: LayoutPreset): Promise<void> => {
      const projectId = getActiveProjectId();
      const requestId = ++latestRequestId;
      await load(preset);

      if (requestId === latestRequestId && projectId === getActiveProjectId() && isCurrent(preset)) {
        apply(preset.id);
      }
    },
    invalidate: (): void => {
      latestRequestId += 1;
    },
  };
};

export const preloadLayoutPresetWidgets = (preset: LayoutPreset): void => {
  for (const region of Object.values(preset.snapshot.widgetRegions)) {
    const typeId = preset.snapshot.widgetInstances[region.activeInstanceId]?.typeId;

    if (typeId) {
      preloadWidget(typeId);
    }
  }
};

export const loadLayoutPresetWidgets = (preset: LayoutPreset): Promise<unknown> =>
  Promise.allSettled(
    Object.values(preset.snapshot.widgetRegions).flatMap((region) => {
      const typeId = preset.snapshot.widgetInstances[region.activeInstanceId]?.typeId;
      const pending = typeId ? loadWidget(typeId) : null;

      return pending ? [pending] : [];
    })
  );
