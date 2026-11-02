import type { AspectRatioPresetsPort } from '@features/generation/contracts';
import type { StoredAspectRatioPreset } from '@workbench/settings/contracts';

import { MAX_ASPECT_RATIO_PRESETS, normalizeAspectRatioPresets } from '@features/generation/settings';
import { createUuid } from '@platform/browser/randomUuid';
import {
  getWorkbenchPreferences,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
} from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const writePresets = (aspectRatioPresets: StoredAspectRatioPreset[]): void => {
  void patchWorkbenchPreferences({ aspectRatioPresets });
};

/**
 * The account's saved aspect ratios, for Generate's Size section (through the Generation UI adapter) and the canvas
 * Frame tool alike. Reads are generation-normalized; writes edit the stored list, so an entry this build cannot use
 * (a newer build's) is kept.
 */
export const useAspectRatioPresets = (): AspectRatioPresetsPort => {
  const { t } = useTranslation();
  const notify = useNotify();
  const stored = useWorkbenchPreferenceSelector((preferences) => preferences.aspectRatioPresets);

  return useMemo<AspectRatioPresetsPort>(() => {
    const presets = normalizeAspectRatioPresets(stored);
    const isFull = presets.length >= MAX_ASPECT_RATIO_PRESETS;
    const read = () => getWorkbenchPreferences().aspectRatioPresets;
    const restore = (preset: StoredAspectRatioPreset, index: number) => {
      const current = read();

      if (current.some((candidate) => candidate.id === preset.id)) {
        return;
      }

      const next = [...current];

      next.splice(Math.min(index, next.length), 0, preset);
      writePresets(next);
    };

    return {
      isFull,
      presets,
      remove: (presetId) => {
        const current = read();
        const index = current.findIndex((candidate) => candidate.id === presetId);
        const preset = current[index];

        if (!preset) {
          return;
        }

        writePresets(current.filter((candidate) => candidate.id !== presetId));
        notify.undoable(t('widgets.generate.aspectRatioPresetDeleted', { name: preset.label.trim() }), {
          onUndo: () => restore(preset, index),
        });
      },
      save: (draft) => {
        if (normalizeAspectRatioPresets(read()).length >= MAX_ASPECT_RATIO_PRESETS) {
          return null;
        }

        const preset = { ...draft, id: createUuid() };

        writePresets([...read(), preset]);
        return preset;
      },
      update: (presetId, draft) => {
        writePresets(read().map((preset) => (preset.id === presetId ? { ...preset, ...draft } : preset)));
      },
    };
  }, [notify, stored, t]);
};
