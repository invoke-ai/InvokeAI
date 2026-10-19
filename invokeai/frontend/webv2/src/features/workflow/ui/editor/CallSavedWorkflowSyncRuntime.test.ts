import type { ParsedWorkflow } from '@features/workflow/core/workflowJson';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  createDeferredCallSavedWorkflowReconciler,
  createSavedWorkflowDocumentParser,
  pruneStaleCallSavedWorkflowNodeState,
} from './CallSavedWorkflowSyncRuntime';

describe('pruneStaleCallSavedWorkflowNodeState', () => {
  it('removes state for nodes no longer in the project graph', () => {
    const state = new Map([
      ['present', 'keep'],
      ['removed', 'drop'],
    ]);

    pruneStaleCallSavedWorkflowNodeState(state, new Set(['present']));

    expect(state).toEqual(new Map([['present', 'keep']]));
  });
});

describe('createDeferredCallSavedWorkflowReconciler', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('defers and coalesces synchronous notifications', () => {
    const reconcile = vi.fn();
    const scheduler = createDeferredCallSavedWorkflowReconciler(reconcile);

    scheduler.schedule();
    scheduler.schedule();

    expect(reconcile).not.toHaveBeenCalled();

    vi.runAllTimers();

    expect(reconcile).toHaveBeenCalledOnce();
  });

  it('cancels pending reconciliation on disposal', () => {
    const reconcile = vi.fn();
    const scheduler = createDeferredCallSavedWorkflowReconciler(reconcile);

    scheduler.schedule();
    scheduler.dispose();
    vi.runAllTimers();

    expect(reconcile).not.toHaveBeenCalled();
  });
});

describe('createSavedWorkflowDocumentParser', () => {
  it('parses each immutable child workflow payload once', () => {
    const parse = vi.fn((workflow: Record<string, unknown>) => ({ document: workflow }) as unknown as ParsedWorkflow);
    const parser = createSavedWorkflowDocumentParser(parse);
    const first = { name: 'first' };
    const replacement = { name: 'replacement' };

    expect(parser(first)).toBe(first);
    expect(parser(first)).toBe(first);
    expect(parser(replacement)).toBe(replacement);
    expect(parse).toHaveBeenCalledTimes(2);
  });

  it('caches malformed payloads without reparsing on every graph update', () => {
    const parse = vi.fn(() => {
      throw new Error('invalid workflow');
    });
    const parser = createSavedWorkflowDocumentParser(parse);
    const malformed = { name: 'malformed' };

    expect(parser(malformed)).toBeUndefined();
    expect(parser(malformed)).toBeUndefined();
    expect(parse).toHaveBeenCalledOnce();
  });

  it('rejects non-object payloads before consulting the object cache', () => {
    const parse = vi.fn();
    const parser = createSavedWorkflowDocumentParser(parse);

    expect(parser(42)).toBeUndefined();
    expect(parser(null)).toBeUndefined();
    expect(parser([])).toBeUndefined();
    expect(parse).not.toHaveBeenCalled();
  });
});
