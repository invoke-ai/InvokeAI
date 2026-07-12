import { describe, expect, it, vi } from 'vitest';

import {
  closeLayerPropertiesForOperation,
  isLayerPropertiesOpen,
  runLayerFilterOperation,
} from './layerPropertiesOperation';

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
