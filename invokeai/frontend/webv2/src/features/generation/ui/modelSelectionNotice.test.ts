import type { GenerateSettings } from '@features/generation/core/types';
import type { TFunction } from 'i18next';

import i18n from 'i18next';
import { beforeAll, describe, expect, it, vi } from 'vitest';

import { describeModelSwitchChanges, notifyGenerateModelSwitched } from './modelSelectionNotice';

const englishCatalog = Object.values(
  import.meta.glob('../../../../public/locales/en.json', { eager: true, import: 'default' })
)[0] as Record<string, unknown>;
const instance = i18n.createInstance();
const t = ((key: string, values?: Record<string, unknown>) => instance.t(key, values)) as TFunction;
const settings = { height: 1600, width: 2848 } as GenerateSettings;

beforeAll(async () => {
  await instance.init({
    initAsync: false,
    interpolation: { escapeValue: false },
    lng: 'en',
    resources: { en: { translation: englishCatalog } },
  });
});

describe('describeModelSwitchChanges', () => {
  it('tells what is lost before what is only set aside, with counts and the new size', () => {
    expect(
      describeModelSwitchChanges(
        {
          changes: [
            { count: 2, kind: 'turnedOff', subject: 'concepts' },
            { kind: 'removed', subject: 'vae' },
            { kind: 'reset', subject: 'clipSkip' },
            { kind: 'removed', subject: 't5EncoderModel' },
            { count: 1, kind: 'turnedOff', subject: 'referenceImages' },
          ],
          dimensionsAdjusted: true,
          settings,
        },
        t,
        'en'
      )
    ).toEqual([
      'Removed: VAE and T5 encoder.',
      'Reset to default: CLIP skip.',
      'Turned off, kept for later: 2 concepts and 1 reference image.',
      'Size set to 2848×1600.',
    ]);
  });

  it('says nothing for a switch that changed nothing else', () => {
    expect(describeModelSwitchChanges({ changes: [], dimensionsAdjusted: false, settings }, t, 'en')).toEqual([]);
  });
});

describe('notifyGenerateModelSwitched', () => {
  const notify = (getCurrentModelKey: () => string | undefined) => {
    const notifications = { info: vi.fn(), undoable: vi.fn() };
    const revert = vi.fn();

    notifyGenerateModelSwitched({
      getCurrentModelKey,
      locale: 'en',
      model: { key: 'flux-dev', name: 'FLUX dev' },
      notifications,
      projectId: 'project-1',
      result: { changes: [{ count: 1, kind: 'turnedOff', subject: 'concepts' }], dimensionsAdjusted: false, settings },
      revert,
      t,
    });

    return { notifications, revert, undo: () => notifications.undoable.mock.calls[0]![1].onUndo() };
  };

  it('reports the switch with an Undo for its project that a newer whole-form change supersedes', () => {
    const { notifications, revert, undo } = notify(() => 'flux-dev');

    expect(notifications.undoable).toHaveBeenCalledExactlyOnceWith('Switched to FLUX dev', {
      message: 'Turned off, kept for later: 1 concept.',
      onUndo: expect.any(Function),
      projectId: 'project-1',
      supersedes: 'generate-settings:project-1',
    });
    undo();
    expect(revert).toHaveBeenCalledOnce();
  });

  it('undoes nothing, and says so, once the project uses another model', () => {
    const { notifications, revert, undo } = notify(() => 'sdxl');

    undo();

    expect(revert).not.toHaveBeenCalled();
    expect(notifications.info).toHaveBeenCalledExactlyOnceWith(
      'The model changed after switching to FLUX dev, so nothing was undone.'
    );
  });
});
