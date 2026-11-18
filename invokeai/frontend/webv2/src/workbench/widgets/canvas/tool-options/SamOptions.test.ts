import type { SamSessionSnapshot } from '@workbench/canvas-operations/operationTypes';

import { describe, expect, it, vi } from 'vitest';

import {
  getSamActionEligibility,
  getSamActionHandlers,
  getSamBusyTranslationKey,
  getSamErrorTranslationKey,
  keepSamImageIntermediate,
  toggleSamSetting,
} from './SamOptions';

const { patchWorkbenchPreferences } = vi.hoisted(() => ({ patchWorkbenchPreferences: vi.fn(() => Promise.resolve()) }));
vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  patchWorkbenchPreferences,
}));

const englishCatalogModules = import.meta.glob('../../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const en = Object.values(englishCatalogModules)[0] as {
  widgets: { layers: { selectObject: Record<string, string> } };
};

const snapshot = (overrides: Partial<SamSessionSnapshot> = {}): SamSessionSnapshot => ({
  applyPolygonRefinement: false,
  autoProcess: false,
  error: null,
  hasPreview: false,
  input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
  invert: false,
  isolatedPreview: true,
  layerName: 'Layer 1',
  layerType: 'raster',
  model: 'segment-anything-2-large',
  pointLabel: 'include',
  retrying: false,
  sourceRect: { height: 20, width: 20, x: 0, y: 0 },
  status: 'ready',
  ...overrides,
});

describe('getSamActionEligibility', () => {
  it('requires valid visual or prompt input before processing', () => {
    expect(getSamActionEligibility(snapshot())).toMatchObject({ canProcess: false });
    expect(
      getSamActionEligibility(
        snapshot({ input: { bbox: null, excludePoints: [], includePoints: [{ x: 1, y: 2 }], type: 'visual' } })
      )
    ).toMatchObject({ canProcess: true });
    expect(getSamActionEligibility(snapshot({ input: { prompt: '  object  ', type: 'prompt' } }))).toMatchObject({
      canProcess: true,
    });
  });

  it('permits Apply and Save only for a current ready preview', () => {
    expect(getSamActionEligibility(snapshot({ hasPreview: true }))).toMatchObject({ canApply: true, canSave: true });
    expect(getSamActionEligibility(snapshot({ hasPreview: true, status: 'processing-sam' }))).toMatchObject({
      canApply: false,
      canProcess: false,
      canSave: false,
    });
    expect(getSamActionEligibility(snapshot({ hasPreview: false }))).toMatchObject({
      canApply: false,
      canSave: false,
    });
    expect(
      getSamActionEligibility(
        snapshot({ error: { code: 'commit-failed', detail: 'commit failed' }, hasPreview: true, status: 'error' })
      )
    ).toMatchObject({
      canApply: true,
      canSave: true,
    });
  });

  it('keeps Cancel available while processing and Reset available for every active session', () => {
    expect(getSamActionEligibility(snapshot({ status: 'processing-sam' }))).toMatchObject({
      canCancel: true,
      canReset: true,
    });
    expect(getSamActionEligibility(snapshot({ error: { code: 'not-ready' }, status: 'error' }))).toMatchObject({
      canReset: true,
    });
    expect(getSamActionEligibility(snapshot({ status: 'ready' }))).toMatchObject({ canReset: true });
  });

  it('disables every mutating control while committing and keeps Cancel available', () => {
    const committing = snapshot({ hasPreview: true, status: 'committing' as never });

    expect(getSamActionEligibility(committing)).toEqual({
      canApply: false,
      canCancel: true,
      canEditInputs: false,
      canProcess: false,
      canReset: false,
      canSave: false,
    });
  });
});

describe('Select Object status copy', () => {
  it.each(['scheduled', 'preparing-source', 'uploading', 'processing-sam', 'rendering-preview', 'committing'] as const)(
    'names the %s phase in localized copy without exposing the identifier',
    (status) => {
      const key = getSamBusyTranslationKey(status)?.split('.').at(-1);
      const copy = key ? en.widgets.layers.selectObject[key] : undefined;
      expect(copy).toBeTypeOf('string');
      expect(copy).not.toContain(status);
    }
  );

  it.each(['ready', 'error'] as const)('shows no phase while %s', (status) => {
    expect(getSamBusyTranslationKey(status)).toBeNull();
  });

  it.each([
    'invalid',
    'not-ready',
    'empty',
    'upload',
    'queue',
    'no-output',
    'reconcile',
    'output-dimension',
    'decode',
    'locked',
    'commit-failed',
    'unknown',
  ] as const)('maps %s errors to localized primary copy', (code) => {
    const key = getSamErrorTranslationKey(code).split('.').at(-1);
    const copy = key ? en.widgets.layers.selectObject[key] : undefined;
    expect(copy).toBeTypeOf('string');
    expect(copy).not.toBe(code);
  });
});

describe('getSamActionHandlers', () => {
  it('keeps adopted SAM images intermediate', async () => {
    await expect(keepSamImageIntermediate('sam-selection.png')).resolves.toBeUndefined();
  });

  it('wires every bar action and save target to the engine', () => {
    const operations = {
      applySelectObjectSession: vi.fn(),
      cancelSelectObjectSession: vi.fn(),
      processSelectObjectSession: vi.fn(),
      resetSelectObjectSession: vi.fn(),
      saveSelectObjectSession: vi.fn(),
    };
    const actions = getSamActionHandlers(operations as never);

    actions.process();
    actions.reset();
    actions.apply();
    actions.save('raster');
    actions.save('control');
    actions.save('inpaint_mask');
    actions.save('regional_guidance');
    actions.save('selection');
    actions.cancel();

    expect(operations.processSelectObjectSession).toHaveBeenCalledOnce();
    expect(operations.resetSelectObjectSession).toHaveBeenCalledOnce();
    expect(operations.applySelectObjectSession).toHaveBeenCalledOnce();
    expect(operations.applySelectObjectSession).toHaveBeenCalledWith(keepSamImageIntermediate);
    expect(operations.processSelectObjectSession.mock.invocationCallOrder[0]).toBeLessThan(
      operations.resetSelectObjectSession.mock.invocationCallOrder[0] as number
    );
    expect(operations.resetSelectObjectSession.mock.invocationCallOrder[0]).toBeLessThan(
      operations.applySelectObjectSession.mock.invocationCallOrder[0] as number
    );
    expect(operations.saveSelectObjectSession).toHaveBeenCalledWith('raster', keepSamImageIntermediate);
    expect(operations.saveSelectObjectSession).toHaveBeenCalledWith('control', keepSamImageIntermediate);
    expect(operations.saveSelectObjectSession).toHaveBeenCalledWith('inpaint_mask', keepSamImageIntermediate);
    expect(operations.saveSelectObjectSession).toHaveBeenCalledWith('regional_guidance', keepSamImageIntermediate);
    expect(operations.saveSelectObjectSession).toHaveBeenCalledWith('selection', keepSamImageIntermediate);
    expect(operations.cancelSelectObjectSession).toHaveBeenCalledOnce();
  });
});

describe('toggleSamSetting', () => {
  it('saves an applied auto-update change as the account preference, and nothing else', () => {
    const updateSelectObjectSession = vi.fn(() => 'updated' as const);

    toggleSamSetting({ updateSelectObjectSession }, 'autoProcess', true);
    toggleSamSetting({ updateSelectObjectSession }, 'isolatedPreview', false);

    expect(updateSelectObjectSession).toHaveBeenNthCalledWith(1, { autoProcess: true });
    expect(updateSelectObjectSession).toHaveBeenNthCalledWith(2, { isolatedPreview: false });
    expect(patchWorkbenchPreferences).toHaveBeenCalledExactlyOnceWith({ selectObjectAutoUpdate: true });
  });

  it('keeps the preference when the session refuses the change', () => {
    patchWorkbenchPreferences.mockClear();

    toggleSamSetting({ updateSelectObjectSession: () => 'stale' }, 'autoProcess', false);

    expect(patchWorkbenchPreferences).not.toHaveBeenCalled();
  });
});
