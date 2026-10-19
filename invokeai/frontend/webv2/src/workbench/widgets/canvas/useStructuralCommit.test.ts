import type { CanvasLayerConfigPatch, StructuralCommitResult } from '@workbench/canvas-engine/api';
import type { TFunction } from 'i18next';

import { createDocumentModel } from '@workbench/canvas-engine/api';
import { createStructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import { layerContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { describe, expect, it, vi } from 'vitest';

import {
  baselineConfig,
  baselinePatch,
  commitPreparedEdit,
  reportPreparedCommit,
  reportStructuralCommit,
} from './useStructuralCommit';

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

  it('commits a prepared edit through the engine transaction, ending any preview before reading the model', () => {
    const order: string[] = [];
    const commitPrepared = vi.fn(() => ({ status: 'committed' as const }));
    const engine = {
      document: {
        model: () => {
          order.push('model');
          return model;
        },
      },
      layers: { commitPrepared, endStructuralPreview: () => void order.push('end-preview') },
    };

    expect(
      commitPreparedEdit(engine, 'Rename', (m) => m.prepare({ id: 'a', patch: { name: 'B' }, type: 'patch' }))
    ).toEqual({
      status: 'committed',
    });
    expect(commitPrepared).toHaveBeenCalledWith(
      'Rename',
      expect.objectContaining({ expectedRevision: 3, projectId: 'p' })
    );
    expect(order).toEqual(['end-preview', 'model']);
  });

  it('never dispatches a refusal or an unchanged command', () => {
    const commitPrepared = vi.fn(() => ({ status: 'committed' as const }));
    const engine = { document: { model: () => model }, layers: { commitPrepared, endStructuralPreview: vi.fn() } };

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
    const engine = {
      document: { model: () => null },
      layers: { commitPrepared: vi.fn(), endStructuralPreview: vi.fn() },
    };
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

describe('baseline narrowing', () => {
  it('records from the baseline fields the edit names, so a session that previewed more fields still commits', () => {
    const stub = createStructuralEngineStub({ layers: [layerContract('c', 'control')] });
    const session = stub.controller.beginPreview()!;
    session.apply({
      config: { adapter: { weight: 0.5 }, layerType: 'control' },
      id: 'c',
      type: 'updateCanvasLayerConfig',
    });
    session.apply({
      config: { adapter: { beginEndStepPct: [0.2, 0.8] }, layerType: 'control' },
      id: 'c',
      type: 'updateCanvasLayerConfig',
    });
    const config: CanvasLayerConfigPatch = { adapter: { beginEndStepPct: [0.2, 0.9] }, layerType: 'control' };

    const before = baselineConfig(session.baseline(), config);
    expect(before).toEqual({ adapter: { beginEndStepPct: [0, 1] }, layerType: 'control' });
    const result = stub.engine.document.model().prepare({ before, config, id: 'c', type: 'patch-config' });
    expect(result.status).toBe('prepared');
    if (result.status !== 'prepared') {
      throw new Error(result.status);
    }
    expect(result.edit.inverse).toMatchObject({ config: { adapter: { beginEndStepPct: [0, 1] } } });
  });

  it('yields nothing for a field the gesture never previewed, leaving the inverse to the live document', () => {
    const baseline = { id: 'c', patch: { name: 'c', opacity: 1 }, type: 'updateCanvasLayer' } as const;

    expect(baselinePatch(baseline, { opacity: 0.5 })).toEqual({ opacity: 1 });
    expect(baselinePatch(baseline, { blendMode: 'screen' })).toBeUndefined();
    expect(baselinePatch(null, { opacity: 0.5 })).toBeUndefined();
    expect(
      baselineConfig(
        { config: { adapter: { weight: 1 }, layerType: 'control' }, id: 'c', type: 'updateCanvasLayerConfig' },
        { adapter: { model: null }, layerType: 'control' }
      )
    ).toBeUndefined();
  });
});
