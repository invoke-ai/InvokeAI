import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as toastsModule from './projectFileToasts';

/**
 * What a project file says while it runs and when it stops.
 *
 * The two properties worth pinning are that it is *one* toast for the whole
 * operation — a slow export must not stack a line per asset — and that a run
 * which lost assets does not finish looking like a clean one.
 */

interface ToastOptions {
  description?: string;
  duration?: number;
  title?: string;
  type?: string;
}

const toaster = vi.hoisted(() => ({
  create: vi.fn((_options: ToastOptions): string => 'toast-1'),
  dismiss: vi.fn((_id: string) => undefined),
  update: vi.fn((_id: string, _options: ToastOptions) => undefined),
}));

vi.mock('@platform/ui', () => ({ toaster }));

let toasts: typeof toastsModule;

/** Stands in for i18next: echoes the key with its interpolations, so both are assertable. */
const t = (key: string, options?: Record<string, unknown>): string =>
  options === undefined ? key : `${key}(${JSON.stringify(options)})`;

beforeEach(async () => {
  vi.resetModules();
  vi.clearAllMocks();
  toasts = await import('./projectFileToasts');
});

describe('startProjectFileReport', () => {
  it('opens one toast that cannot expire while the transfer runs', () => {
    toasts.startProjectFileReport(t, 'projects.exporting');

    expect(toaster.create).toHaveBeenCalledTimes(1);
    expect(toaster.create.mock.calls[0]![0]).toMatchObject({
      duration: Number.POSITIVE_INFINITY,
      title: 'projects.exporting',
      type: 'loading',
    });
  });

  it('updates that one toast rather than creating another per step', () => {
    const report = toasts.startProjectFileReport(t, 'projects.exporting');

    report.report({ completed: 1, phase: 'bundling', total: 3 });
    report.report({ completed: 2, phase: 'bundling', total: 3 });
    report.succeed('projects.exported', []);

    expect(toaster.create).toHaveBeenCalledTimes(1);
    expect(toaster.update).toHaveBeenCalledTimes(3);
    expect(toaster.update.mock.calls.every(([id]) => id === 'toast-1')).toBe(true);
  });

  it('counts assets while bundling and stops counting while packing', () => {
    const report = toasts.startProjectFileReport(t, 'projects.exporting');

    report.report({ completed: 2, phase: 'bundling', total: 7 });
    expect(toaster.update.mock.calls[0]![1].description).toBe(
      'projects.file.bundlingProgress({"completed":2,"total":7})'
    );

    report.report({ completed: 7, phase: 'packing', total: 7 });
    expect(toaster.update.mock.calls[1]![1].description).toBe('projects.file.packing');
  });

  it('counts uploads on the way back in', () => {
    const report = toasts.startProjectFileReport(t, 'projects.importing');

    report.report({ completed: 1, phase: 'restoring', total: 4 });

    expect(toaster.update.mock.calls[0]![1].description).toBe(
      'projects.file.restoringProgress({"completed":1,"total":4})'
    );
  });

  it('finishes a clean run as a plain success with nothing to explain', () => {
    const report = toasts.startProjectFileReport(t, 'projects.exporting');

    report.succeed('projects.exported', []);

    expect(toaster.update).toHaveBeenCalledWith('toast-1', {
      duration: undefined,
      title: 'projects.exported',
      type: 'success',
    });
  });

  /** A project that lost forty layers must not read the same as one that lost none. */
  it('finishes a lossy run as a warning naming the count', () => {
    const report = toasts.startProjectFileReport(t, 'projects.exporting');

    report.succeed('projects.exported', ['a.png', 'b.png']);

    expect(toaster.update).toHaveBeenCalledWith('toast-1', {
      description: 'projects.file.missingAssets({"count":2})',
      duration: undefined,
      title: 'projects.exported',
      type: 'warning',
    });
  });

  it('translates a format error into its reason rather than its message', async () => {
    const { InvkFormatError } = await import('./invk/format');
    const report = toasts.startProjectFileReport(t, 'projects.importing');

    report.fail('projects.importFailed', new InvkFormatError('legacy-canvas-project'));

    expect(toaster.update).toHaveBeenCalledWith('toast-1', {
      description: 'projects.file.legacyCanvasProject',
      duration: undefined,
      title: 'projects.importFailed',
      type: 'error',
    });
  });

  it('passes through the message of an error that is not ours', () => {
    const report = toasts.startProjectFileReport(t, 'projects.importing');

    report.fail('projects.importFailed', new Error('The server is on fire.'));

    expect(toaster.update.mock.calls[0]![1]).toMatchObject({
      description: 'The server is on fire.',
      type: 'error',
    });
  });

  it('takes the toast down without a verdict when there is no one left to tell', () => {
    const report = toasts.startProjectFileReport(t, 'projects.exporting');

    report.dismiss();

    expect(toaster.dismiss).toHaveBeenCalledWith('toast-1');
    expect(toaster.update).not.toHaveBeenCalled();
  });
});
