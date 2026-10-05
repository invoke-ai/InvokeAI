import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as toastsModule from './projectFileToasts';

/** Update one toast throughout the transfer; partial loss finishes as a warning. */

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
    report.succeed('projects.exported', { boardItemIssues: [], documentReferenceIssues: [] });

    expect(toaster.create).toHaveBeenCalledTimes(1);
    expect(toaster.update.mock.calls.every(([id]) => id === 'toast-1')).toBe(true);
  });

  /** Show the first progress update immediately and coalesce subsequent per-asset updates. */
  it('redraws at most once per interval while assets settle', () => {
    vi.useFakeTimers();

    try {
      const report = toasts.startProjectFileReport(t, 'projects.exporting');

      report.report({ completed: 1, phase: 'bundling', total: 300 });
      report.report({ completed: 2, phase: 'bundling', total: 300 });
      report.report({ completed: 3, phase: 'bundling', total: 300 });

      expect(toaster.update).toHaveBeenCalledTimes(1);
      expect(toaster.update.mock.calls[0]![1].description).toBe(
        'projects.file.bundlingProgress({"completed":1,"total":300})'
      );

      // Whatever the latest count is when the interval expires — never a stale one.
      vi.advanceTimersByTime(500);
      expect(toaster.update).toHaveBeenCalledTimes(2);
      expect(toaster.update.mock.calls[1]![1].description).toBe(
        'projects.file.bundlingProgress({"completed":3,"total":300})'
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it('counts assets while bundling and stops counting while packing', () => {
    vi.useFakeTimers();

    try {
      const report = toasts.startProjectFileReport(t, 'projects.exporting');

      report.report({ completed: 2, phase: 'bundling', total: 7 });
      expect(toaster.update.mock.calls[0]![1].description).toBe(
        'projects.file.bundlingProgress({"completed":2,"total":7})'
      );

      vi.advanceTimersByTime(500);
      report.report({ completed: 7, phase: 'packing', total: 7 });
      expect(toaster.update.mock.calls[1]![1].description).toBe('projects.file.packing');
    } finally {
      vi.useRealTimers();
    }
  });

  /** Navigation failure after transfer cannot change its successful verdict. */
  it('does not take back a verdict it has already given', () => {
    const report = toasts.startProjectFileReport(t, 'projects.importing');

    report.succeed('projects.imported', { boardItemIssues: [], documentReferenceIssues: [] });
    report.fail('projects.importFailed', new Error('navigation blew up'));

    const settles = toaster.update.mock.calls.filter(([, options]) => options.duration === undefined);

    expect(settles).toHaveLength(1);
    expect(settles[0]![1]).toMatchObject({ title: 'projects.imported', type: 'success' });
  });

  it('counts uploads on the way back in', () => {
    const report = toasts.startProjectFileReport(t, 'projects.importing');

    report.report({ completed: 1, phase: 'restoring', total: 4 });

    expect(toaster.update.mock.calls[0]![1].description).toBe(
      'projects.file.restoringProgress({"completed":1,"total":4})'
    );
  });

  /** Count missing board media separately from missing document references. */
  it.each([
    ['nothing', [], [], { title: 'projects.exported', type: 'success' }],
    [
      'only document references',
      [],
      ['a.png', 'b.png'],
      { description: 'projects.file.missingReferences({"count":2})', title: 'projects.exported', type: 'warning' },
    ],
    [
      'only board items',
      ['unreferenced.png'],
      [],
      { description: 'projects.file.missingBoardItems({"count":1})', title: 'projects.exported', type: 'warning' },
    ],
    [
      'both, counted apart',
      ['clip.mp4'],
      ['a.png', 'b.png'],
      {
        description: 'projects.file.missingBoardItems({"count":1}) projects.file.missingReferences({"count":2})',
        title: 'projects.exported',
        type: 'warning',
      },
    ],
  ])('settles a run that lost %s', (_label, boardNames, referenceNames, expected) => {
    const issue = (name: string) => ({ kind: 'image' as const, name, reason: 'fetch-failed' as const });
    const report = toasts.startProjectFileReport(t, 'projects.exporting');

    report.succeed('projects.exported', {
      boardItemIssues: boardNames.map(issue),
      documentReferenceIssues: referenceNames.map(issue),
    });

    expect(toaster.update).toHaveBeenCalledWith('toast-1', { duration: undefined, ...expected });
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
