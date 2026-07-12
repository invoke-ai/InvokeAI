import type { SamSessionSnapshot } from '@workbench/canvas-engine/engineStores';

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import {
  getSamActionEligibility,
  getSamErrorTranslationKey,
  getSamStatusTranslationKey,
  SamProcessFeedback,
} from './SamOptions';

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
      getSamActionEligibility(
        snapshot({ error: { code: 'unknown', detail: 'commit failed' }, hasPreview: true, status: 'error' })
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
    'unknown',
  ] as const)('maps %s errors to localized primary copy', (code) => {
    const key = getSamErrorTranslationKey(code).split('.').at(-1);
    const copy = key ? en.widgets.layers.selectObject[key] : undefined;
    expect(copy).toBeTypeOf('string');
    expect(copy).not.toBe(code);
  });

  it('server-renders a localized phase with polite status semantics', () => {
    const markup = renderToStaticMarkup(
      createElement(SamProcessFeedback, {
        error: null,
        errorText: null,
        statusText: 'Rendering object preview…',
      })
    );

    expect(markup).toContain('role="status"');
    expect(markup).toContain('aria-live="polite"');
    expect(markup).toContain('Rendering object preview…');
    expect(markup).not.toContain('role="alert"');
  });

  it('server-renders a known localized error as the assertive primary message', () => {
    const markup = renderToStaticMarkup(
      createElement(SamProcessFeedback, {
        error: { code: 'decode' },
        errorText: 'The object preview could not be decoded.',
        statusText: 'Select Object needs attention.',
      })
    );

    expect(markup).toContain('role="alert"');
    expect(markup).toContain('aria-live="assertive"');
    expect(markup).toContain('The object preview could not be decoded.');
  });

  it('server-renders unknown diagnostics only after a localized primary message', () => {
    const markup = renderToStaticMarkup(
      createElement(SamProcessFeedback, {
        error: { code: 'unknown', detail: 'GPU worker disconnected' },
        errorText: 'Select Object could not finish.',
        statusText: 'Select Object needs attention.',
      })
    );

    expect(markup.indexOf('Select Object could not finish.')).toBeLessThan(markup.indexOf('GPU worker disconnected'));
    expect(markup).toContain('role="alert"');
  });
});
