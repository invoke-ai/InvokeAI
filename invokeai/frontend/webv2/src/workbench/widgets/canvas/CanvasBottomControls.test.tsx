import type { ReactNode } from 'react';

import {
  getLayerPropertiesRequest,
  requestLayerProperties,
} from '@workbench/widgets/layers/layerPropertiesRequestStore';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';

import {
  CanvasBottomControlsPresentation,
  clearLayerPropertiesForOperationPresentation,
  resolveBottomControlSlots,
} from './CanvasBottomControls';

const REGULAR_CONTENT = <span>regular</span>;
const renderOperation = vi.fn((locked: boolean): ReactNode => <span data-locked={locked}>operation</span>);

describe('canvas bottom controls', () => {
  it('consumes any programmatic layer-properties request at the operation presentation boundary', () => {
    requestLayerProperties('other-layer', 'filter');

    clearLayerPropertiesForOperationPresentation();

    expect(getLayerPropertiesRequest()).toBeNull();
  });

  it.each([
    { externalLock: false, operationKind: null, expected: { operation: false, regular: true }, scenario: 'idle' },
    { externalLock: true, operationKind: null, expected: { operation: false, regular: false }, scenario: 'staging' },
    { externalLock: true, operationKind: null, expected: { operation: false, regular: false }, scenario: 'generation' },
    {
      externalLock: false,
      operationKind: 'filter' as const,
      expected: { operation: true, regular: false },
      scenario: 'active Filter',
    },
    {
      externalLock: true,
      operationKind: 'filter' as const,
      expected: { operation: true, regular: false },
      scenario: 'active Filter plus staging and temporary View',
    },
    {
      externalLock: false,
      operationKind: 'select-object' as const,
      expected: { operation: true, regular: false },
      scenario: 'active SAM',
    },
    {
      externalLock: true,
      operationKind: 'select-object' as const,
      expected: { operation: true, regular: false },
      scenario: 'active SAM plus generation and temporary View',
    },
  ])('resolves $scenario bottom controls', ({ expected, externalLock, operationKind }) => {
    expect(resolveBottomControlSlots({ isExternalInteractionLocked: externalLock, operationKind })).toEqual(expected);
  });

  it('mounts active operation content while locked, forwards the lock, and hides regular content', () => {
    renderOperation.mockClear();
    const output = renderToStaticMarkup(
      <CanvasBottomControlsPresentation
        isExternalInteractionLocked
        operationKind="filter"
        regularContent={REGULAR_CONTENT}
        renderOperation={renderOperation}
      />
    );

    expect(output).toContain('data-locked="true"');
    expect(output).toContain('operation');
    expect(output).not.toContain('regular');
    expect(renderOperation).toHaveBeenCalledWith(true);
  });

  it.each([
    { externalLock: false, expected: '<span>regular</span>', operationKind: null },
    { externalLock: true, expected: '', operationKind: null },
  ])('mounts idle content for externalLock=$externalLock', ({ expected, externalLock, operationKind }) => {
    expect(
      renderToStaticMarkup(
        <CanvasBottomControlsPresentation
          isExternalInteractionLocked={externalLock}
          operationKind={operationKind}
          regularContent={REGULAR_CONTENT}
          renderOperation={renderOperation}
        />
      )
    ).toBe(expected);
  });
});
