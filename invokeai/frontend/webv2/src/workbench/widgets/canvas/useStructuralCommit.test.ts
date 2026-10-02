import type { StructuralCommitResult } from '@workbench/canvas-engine/api';
import type { TFunction } from 'i18next';

import { createDocumentModel } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { describe, expect, it, vi } from 'vitest';

import { commitPreparedEdit, reportPreparedCommit, reportStructuralCommit } from './useStructuralCommit';

const t = ((key: string) => key) as unknown as TFunction;

describe('reportStructuralCommit', () => {
  it.each<StructuralCommitResult>([{ status: 'committed' }, { status: 'busy' }])('stays silent for %j', (result) => {
    const report = vi.fn();

    reportStructuralCommit(result, report, t);

    expect(report).not.toHaveBeenCalled();
  });

  it.each<[StructuralCommitResult, string]>([
    [{ status: 'gesture-active' }, 'widgets.canvas.structural.gestureActive'],
    [{ status: 'not-ready' }, 'widgets.canvas.structural.notReady'],
    [{ actualRevision: 2, expectedRevision: 1, status: 'stale' }, 'widgets.canvas.structural.stale'],
    [{ status: 'dispatch-rejected' }, 'widgets.canvas.structural.rejected'],
    [{ recovered: 'reverted', status: 'postcondition-failed' }, 'widgets.canvas.structural.reverted'],
    [{ recovered: 'reverted-unmirrored', status: 'postcondition-failed' }, 'widgets.canvas.structural.unmirrored'],
    [{ recovered: 'unreverted', status: 'postcondition-failed' }, 'widgets.canvas.structural.unverified'],
  ])('explains %j', (result, message) => {
    const report = vi.fn();

    reportStructuralCommit(result, report, t);

    expect(report).toHaveBeenCalledWith('widgets.canvas.structural.failed', message);
  });
});

describe('commitPreparedEdit', () => {
  const layer = createEmptyPaintLayer('Layer', 'a');
  const document = { ...createEmptyCanvasDocument(), stacks: stacksFrom([layer]), selectedLayerId: 'a' };
  const model = createDocumentModel(document, { editRevision: 3, projectId: 'p' });

  it('commits a prepared edit through the engine transaction', () => {
    const commitPrepared = vi.fn(() => ({ status: 'committed' as const }));
    const engine = { document: { model: () => model }, layers: { commitPrepared } };

    expect(
      commitPreparedEdit(engine, 'Rename', (m) => m.prepare({ id: 'a', patch: { name: 'B' }, type: 'patch' }))
    ).toEqual({
      status: 'committed',
    });
    expect(commitPrepared).toHaveBeenCalledWith(
      'Rename',
      expect.objectContaining({ expectedRevision: 3, projectId: 'p' })
    );
  });

  it('never dispatches a refusal or an unchanged command', () => {
    const commitPrepared = vi.fn(() => ({ status: 'committed' as const }));
    const engine = { document: { model: () => model }, layers: { commitPrepared } };

    expect(commitPreparedEdit(engine, 'Delete', (m) => m.prepare({ ids: ['ghost'], type: 'remove' }))).toEqual({
      refusal: { ids: ['ghost'], status: 'missing' },
      status: 'refused',
    });
    expect(commitPreparedEdit(engine, 'Select', (m) => m.prepare({ id: 'a', type: 'select' }))).toEqual({
      status: 'unchanged',
    });
    expect(commitPrepared).not.toHaveBeenCalled();
  });

  it('refuses as not-ready without an engine or a document', () => {
    expect(commitPreparedEdit(null, 'Rename', () => ({ status: 'unchanged' }))).toEqual({ status: 'not-ready' });
    const engine = { document: { model: () => null }, layers: { commitPrepared: vi.fn() } };
    expect(commitPreparedEdit(engine, 'Rename', () => ({ status: 'unchanged' }))).toEqual({ status: 'not-ready' });
  });
});

describe('reportPreparedCommit', () => {
  const t = ((key: string) => key) as TFunction;

  it('stays silent for unchanged and reports refusals with their own message', () => {
    const reportError = vi.fn();
    reportPreparedCommit({ status: 'unchanged' }, reportError, t);
    expect(reportError).not.toHaveBeenCalled();
    reportPreparedCommit({ refusal: { ids: ['a'], status: 'locked' }, status: 'refused' }, reportError, t);
    expect(reportError).toHaveBeenLastCalledWith(
      'widgets.canvas.structural.failed',
      'widgets.canvas.structural.refusedLocked'
    );
  });

  it('forwards transaction outcomes to the structural reporter', () => {
    const reportError = vi.fn();
    reportPreparedCommit({ status: 'stale', actualRevision: 2, expectedRevision: 1 }, reportError, t);
    expect(reportError).toHaveBeenCalledWith('widgets.canvas.structural.failed', 'widgets.canvas.structural.stale');
  });
});
