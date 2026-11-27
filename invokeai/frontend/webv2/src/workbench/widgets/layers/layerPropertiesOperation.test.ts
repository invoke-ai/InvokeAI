import { describe, expect, it, vi } from 'vitest';

import {
  closeLayerPropertiesForOperation,
  getLayerFilterLaunchDisabledReason,
  isLayerPropertiesOpen,
  runLayerFilterOperation,
} from './layerPropertiesOperation';

describe('getLayerFilterLaunchDisabledReason', () => {
  it.each([
    {
      expected: 'not-ready',
      input: { hasEngine: false, hasExportableContent: false, isEnabled: true, isLocked: false },
      scenario: 'missing engine',
    },
    {
      expected: 'disabled',
      input: { hasEngine: true, hasExportableContent: true, isEnabled: false, isLocked: false },
      scenario: 'disabled layer',
    },
    {
      expected: 'locked',
      input: { hasEngine: true, hasExportableContent: true, isEnabled: true, isLocked: true },
      scenario: 'locked layer',
    },
    {
      expected: 'empty',
      input: { hasEngine: true, hasExportableContent: false, isEnabled: true, isLocked: false },
      scenario: 'empty layer',
    },
    {
      expected: null,
      input: { hasEngine: true, hasExportableContent: true, isEnabled: true, isLocked: false },
      scenario: 'eligible layer',
    },
  ] as const)('returns $expected for $scenario', ({ expected, input }) => {
    expect(getLayerFilterLaunchDisabledReason(input)).toBe(expected);
  });
});

describe('runLayerFilterOperation callbacks', () => {
  it('notifies only the success callback for a started launch', () => {
    const onOperationStarted = vi.fn();
    const onOperationRejected = vi.fn();

    expect(runLayerFilterOperation(() => 'started', onOperationStarted, onOperationRejected)).toBe('started');
    expect(onOperationStarted).toHaveBeenCalledOnce();
    expect(onOperationRejected).not.toHaveBeenCalled();
  });

  it.each(['missing', 'disabled', 'locked', 'unsupported', 'not-ready'] as const)(
    'notifies rejection for %s without closing properties',
    (result) => {
      const onOperationStarted = vi.fn();
      const onOperationRejected = vi.fn();

      expect(runLayerFilterOperation(() => result, onOperationStarted, onOperationRejected)).toBe(result);
      expect(onOperationStarted).not.toHaveBeenCalled();
      expect(onOperationRejected).toHaveBeenCalledOnce();
      expect(onOperationRejected).toHaveBeenCalledWith(result);
    }
  );

  it('does not report rejection when no engine supplied a launch result', () => {
    const onOperationStarted = vi.fn();
    const onOperationRejected = vi.fn();

    expect(runLayerFilterOperation(() => undefined, onOperationStarted, onOperationRejected)).toBeUndefined();
    expect(onOperationStarted).not.toHaveBeenCalled();
    expect(onOperationRejected).not.toHaveBeenCalled();
  });
});

describe('closeLayerPropertiesForOperation', () => {
  it.each([
    { requestToken: null, triggerOpen: true },
    { requestToken: 42, triggerOpen: false },
    { requestToken: 42, triggerOpen: true },
  ])('clears trigger and request ownership for $triggerOpen/$requestToken', (state) => {
    const closed = closeLayerPropertiesForOperation(state);

    expect(closed).toEqual({ requestToken: null, requestTokenToClear: state.requestToken, triggerOpen: false });
    expect(isLayerPropertiesOpen(closed)).toBe(false);
  });
});
