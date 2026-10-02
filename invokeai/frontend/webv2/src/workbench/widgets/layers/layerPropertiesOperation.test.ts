import { describe, expect, it, vi } from 'vitest';

import {
  closeLayerPropertiesForOperation,
  getLayerFilterLaunchDisabledReason,
  getLayerFilterLaunchReasonKey,
  getLayerPropertiesOwnershipKey,
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

  it.each([
    ['empty', 'widgets.layers.actions.empty'],
    ['missing', 'widgets.layers.actions.missing'],
    ['disabled', 'widgets.layers.actions.disabled'],
    ['locked', 'widgets.layers.actions.locked'],
    ['unsupported', 'widgets.layers.actions.unsupported'],
    ['not-ready', 'widgets.layers.actions.notReady'],
  ] as const)('maps %s to %s', (reason, key) => {
    expect(getLayerFilterLaunchReasonKey(reason)).toBe(key);
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
  it('uses separate ownership across editing-lock transitions so trigger state cannot reopen after cancel', () => {
    expect(getLayerPropertiesOwnershipKey(false)).not.toBe(getLayerPropertiesOwnershipKey(true));
  });

  it.each([
    { requestToken: null, triggerOpen: true },
    { requestToken: 42, triggerOpen: false },
    { requestToken: 42, triggerOpen: true },
  ])('clears trigger and request ownership for $triggerOpen/$requestToken', (state) => {
    const closed = closeLayerPropertiesForOperation(state);

    expect(closed).toEqual({ requestToken: null, requestTokenToClear: state.requestToken, triggerOpen: false });
    expect(isLayerPropertiesOpen(closed)).toBe(false);
  });

  it('stays closed after the operation is later canceled', () => {
    const closed = closeLayerPropertiesForOperation({ requestToken: 7, triggerOpen: true });

    expect(isLayerPropertiesOpen(closed)).toBe(false);
  });
});

describe('runLayerFilterOperation', () => {
  it('notifies the popover after a successful manual launch', () => {
    const onOperationStarted = vi.fn();

    expect(runLayerFilterOperation(() => 'started', onOperationStarted)).toBe('started');
    expect(onOperationStarted).toHaveBeenCalledOnce();
  });

  it('notifies the popover after a successful model-recommendation launch', () => {
    const onOperationStarted = vi.fn();
    const start = vi.fn((_recommendation: string) => 'started' as const);

    expect(runLayerFilterOperation(() => start('normal_map'), onOperationStarted)).toBe('started');
    expect(start).toHaveBeenCalledWith('normal_map');
    expect(onOperationStarted).toHaveBeenCalledOnce();
  });

  it.each(['locked', 'not-ready', 'missing', undefined] as const)('does not close after a %s launch', (result) => {
    const onOperationStarted = vi.fn();

    expect(runLayerFilterOperation(() => result, onOperationStarted)).toBe(result);
    expect(onOperationStarted).not.toHaveBeenCalled();
  });
});
