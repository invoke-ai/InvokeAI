import type { GenerateReferenceImage, GenerateSettings } from '@features/generation/core/types';

import { moveReferenceImage } from '@features/generation/core/settings';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { createStableSelector } from '@platform/state/selectors';
import { describe, expect, it } from 'vitest';

import {
  applyGenerateSettingsUpdate,
  mergeGenerateSettingsUpdate,
  pickGenerateSettings,
  reuseEqualGenerateSettingsValues,
} from './generateDebounce';

/** Updaters run for both draft and flush; mint IDs outside them. */
const buildSettings = (referenceImages: GenerateReferenceImage[] = []): GenerateSettings =>
  ({ referenceImages }) as unknown as GenerateSettings;

const entry = (id: string): GenerateReferenceImage => ({
  config: { image: null, type: 'external_reference_image' },
  id,
  isEnabled: true,
});

const idsOf = (settings: GenerateSettings) => settings.referenceImages.map(({ id }) => id);

describe('pending generate settings updates', () => {
  it('applies a pending updater twice, so an id minted inside one diverges between draft and commit', () => {
    let minted = 0;
    const appendWithIdInsideUpdater = (settings: GenerateSettings): GenerateSettings => ({
      ...settings,
      referenceImages: [...settings.referenceImages, entry(`minted-${String((minted += 1))}`)],
    });

    const pending = mergeGenerateSettingsUpdate(null, appendWithIdInsideUpdater);
    const draft = applyGenerateSettingsUpdate(buildSettings([entry('a')]), pending);
    const committed = applyGenerateSettingsUpdate(buildSettings([entry('a')]), pending);

    expect(idsOf(draft)).toEqual(['a', 'minted-1']);
    expect(idsOf(committed)).toEqual(['a', 'minted-2']);

    // The rendered ID must match the flushed ID for reordering to work.
    const chained = mergeGenerateSettingsUpdate(pending, (settings) => ({
      ...settings,
      referenceImages: [...moveReferenceImage(settings.referenceImages, 'minted-1', -1)],
    }));

    expect(idsOf(applyGenerateSettingsUpdate(buildSettings([entry('a')]), chained))).toEqual(['a', 'minted-3']);
  });

  it('keeps the reorder when the appended ids are minted outside the updater', () => {
    const ids = ['hoisted'];
    const appendWithHoistedId = (settings: GenerateSettings): GenerateSettings => ({
      ...settings,
      referenceImages: [...settings.referenceImages, entry(ids[0] ?? '')],
    });

    const chained = mergeGenerateSettingsUpdate(mergeGenerateSettingsUpdate(null, appendWithHoistedId), (settings) => ({
      ...settings,
      referenceImages: [...moveReferenceImage(settings.referenceImages, 'hoisted', -1)],
    }));

    expect(idsOf(applyGenerateSettingsUpdate(buildSettings([entry('a')]), chained))).toEqual(['hoisted', 'a']);
    expect(idsOf(applyGenerateSettingsUpdate(buildSettings([entry('a')]), chained))).toEqual(['hoisted', 'a']);
  });
});

describe('draft selection', () => {
  const createDraft = () =>
    createExternalStoreCore({
      cfgScale: 7,
      loras: [],
      seed: 1,
      steps: 30,
    } as unknown as GenerateSettings);
  // The shallow equality the external-store selector hooks apply by default.
  const selectSampling = () => createStableSelector(pickGenerateSettings(['seed', 'steps']));

  it('keeps a selection until one of its own fields changes', () => {
    const draft = createDraft();
    const select = selectSampling();
    const selected = select(draft.getSnapshot());

    expect(selected).toEqual({ seed: 1, steps: 30 });

    draft.patchSnapshot({ cfgScale: 4 });
    expect(select(draft.getSnapshot())).toBe(selected);

    draft.patchSnapshot({ steps: 12 });
    expect(select(draft.getSnapshot())).toEqual({ seed: 1, steps: 12 });
  });

  it('keeps a selection when a reconciled update only re-creates equal values', () => {
    const draft = createDraft();
    const select = createStableSelector(pickGenerateSettings(['loras']));
    const selected = select(draft.getSnapshot());
    // A stored-value round trip re-creates arrays that are structurally unchanged.
    const recreated = { ...draft.getSnapshot(), loras: [] };

    draft.setSnapshot(reuseEqualGenerateSettingsValues(draft.getSnapshot(), recreated));

    expect(select(draft.getSnapshot())).toBe(selected);
  });
});
