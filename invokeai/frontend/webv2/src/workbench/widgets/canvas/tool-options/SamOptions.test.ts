import type { SamSessionSnapshot } from '@workbench/canvas-engine/engineStores';

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { getSamActionEligibility, getSamStatusTranslationKey, SamProcessFeedback } from './SamOptions';

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
  model: 'segment-anything-2-large',
  pointLabel: 'include',
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
      getSamActionEligibility(snapshot({ error: 'commit failed', hasPreview: true, status: 'error' }))
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

  it('disables mutating actions under an external interaction lock but preserves Cancel', () => {
    expect(getSamActionEligibility(snapshot({ hasPreview: true }), true)).toEqual({
      canApply: false,
      canCancel: true,
      canEditInputs: false,
      canProcess: false,
      canReset: false,
      canSave: false,
    });
  });
});

describe('SamProcessFeedback', () => {
  it.each([
    'ready',
    'scheduled',
    'preparing-composite',
    'uploading',
    'processing-sam',
    'rendering-preview',
    'committing',
    'error',
  ] as const)('maps %s to localized user-facing copy without exposing the identifier', (status) => {
    const key = getSamStatusTranslationKey(status).split('.').at(-1);
    const copy = key ? en.widgets.layers.selectObject[key] : undefined;
    expect(copy).toBeTypeOf('string');
    expect(copy).not.toContain(status);
  });

  it('server-renders polite status and assertive error feedback visibly', () => {
    const markup = renderToStaticMarkup(
      createElement(SamProcessFeedback, {
        error: 'The mask could not be decoded.',
        statusText: 'Rendering object preview…',
      })
    );

    expect(markup).toContain('role="status"');
    expect(markup).toContain('aria-live="polite"');
    expect(markup).toContain('Rendering object preview…');
    expect(markup).toContain('role="alert"');
    expect(markup).toContain('aria-live="assertive"');
    expect(markup).toContain('The mask could not be decoded.');
  });
});
