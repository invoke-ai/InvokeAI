import type { LayoutPreset, LayoutPresetId } from './layoutContracts';

import { loadWidget, preloadWidget } from './widgetRegistry';

interface LayoutPresetActivatorDependencies {
  apply: (presetId: LayoutPresetId) => void;
  load: (preset: LayoutPreset) => Promise<unknown>;
}

export const createLayoutPresetActivator = ({ apply, load }: LayoutPresetActivatorDependencies) => {
  let latestRequestId = 0;

  return async (preset: LayoutPreset): Promise<void> => {
    const requestId = ++latestRequestId;
    await load(preset);

    if (requestId === latestRequestId) {
      apply(preset.id);
    }
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
