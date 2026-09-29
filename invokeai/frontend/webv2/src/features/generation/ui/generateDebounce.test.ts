import type { GenerateReferenceImage, GenerateSettings } from '@features/generation/core/types';

import { moveReferenceImage } from '@features/generation/core/settings';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { describe, expect, it } from 'vitest';

import {
  applyGenerateSettingsUpdate,
  createDraftTracker,
  isDraftView,
  mergeGenerateSettingsUpdate,
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

// Identity is compared outside `expect`, whose formatting would enumerate (and so fully track) a view.
describe('draft tracker', () => {
  const createDraft = () => createExternalStoreCore({ cfgScale: 7, seed: 1, steps: 30 } as unknown as GenerateSettings);

  it('keeps the view when only an unread key changes, and reads the latest draft through it', () => {
    const draft = createDraft();
    const getView = createDraftTracker(draft);
    const view = getView();

    expect(view.steps).toBe(30);
    draft.patchSnapshot({ cfgScale: 4 });

    expect(getView() === view).toBe(true);
    // Unread keys are not stale: the retained view reads the latest draft.
    expect(view.cfgScale).toBe(4);

    draft.patchSnapshot({ steps: 12 });

    const next = getView();
    expect(next === view).toBe(false);
    expect(next.steps).toBe(12);
    expect(isDraftView(next)).toBe(true);
    expect(isDraftView(draft.getSnapshot())).toBe(false);
  });

  it('tracks a key read only on a conditional branch', () => {
    const draft = createDraft();
    const getView = createDraftTracker(draft);
    const view = getView();

    if (view.steps > 10) {
      expect(view.seed).toBe(1);
    }

    draft.patchSnapshot({ seed: 2 });

    expect(getView() === view).toBe(false);
  });

  it('treats enumeration as reading every key', () => {
    const draft = createDraft();
    const getView = createDraftTracker(draft);
    const view = getView();

    expect({ ...view }).toEqual({ cfgScale: 7, seed: 1, steps: 30 });
    draft.patchSnapshot({ seed: 2 });

    const next = getView();
    expect(next === view).toBe(false);

    // A fresh view starts with no reads, so an unread change keeps it.
    draft.patchSnapshot({ cfgScale: 3 });
    expect(getView() === next).toBe(true);
  });
});
