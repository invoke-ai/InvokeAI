import type { LayoutPreset, LayoutPresetId } from './layoutContracts';
import type { AccountState } from './projectContracts';

import { layoutPresets } from './layoutPresets';
import { resolveSavedLayoutPreset } from './layoutPresetSnapshots';

/**
 * Preserve saved relative order, inserting unseen presets beside canonical neighbors so newly shipped presets do
 * not always append.
 */
export const normalizeLayoutPresetOrder = (value: unknown, presets: readonly LayoutPreset[]): LayoutPresetId[] => {
  const knownIds = new Set(presets.map(({ id }) => id));
  const seenIds = new Set<LayoutPresetId>();
  const orderedIds: LayoutPresetId[] = [];

  if (Array.isArray(value)) {
    for (const id of value) {
      if (typeof id === 'string' && knownIds.has(id) && !seenIds.has(id)) {
        orderedIds.push(id);
        seenIds.add(id);
      }
    }
  }

  presets.forEach(({ id }, presetIndex) => {
    if (seenIds.has(id)) {
      return;
    }

    let insertAt = 0;

    for (let index = presetIndex - 1; index >= 0; index -= 1) {
      const precedingIndex = orderedIds.indexOf(presets[index]!.id);

      if (precedingIndex >= 0) {
        insertAt = precedingIndex + 1;
        break;
      }
    }

    orderedIds.splice(insertAt, 0, id);
    seenIds.add(id);
  });

  return orderedIds;
};

export const getOrderedLayoutPresets = (account: AccountState): LayoutPreset[] => {
  const availablePresets = [...layoutPresets, ...(account.customLayoutPresets ?? [])].map(({ id }) =>
    resolveSavedLayoutPreset(account, id)
  );
  const presetById = new Map(availablePresets.map((preset) => [preset.id, preset]));

  return normalizeLayoutPresetOrder(account.layoutPresetOrder, availablePresets).flatMap((id) => {
    const preset = presetById.get(id);

    return preset ? [preset] : [];
  });
};

export const reorderLayoutPresetIds = (
  account: AccountState,
  activeId: LayoutPresetId,
  overId: LayoutPresetId
): LayoutPresetId[] | null => {
  if (activeId === overId) {
    return null;
  }

  const orderedIds = getOrderedLayoutPresets(account).map(({ id }) => id);
  const activeIndex = orderedIds.indexOf(activeId);
  const overIndex = orderedIds.indexOf(overId);

  if (activeIndex < 0 || overIndex < 0) {
    return null;
  }

  const nextIds = [...orderedIds];
  const [movedId] = nextIds.splice(activeIndex, 1);
  nextIds.splice(overIndex, 0, movedId);

  return nextIds;
};
