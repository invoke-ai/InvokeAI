import type { CanvasEngine, StartFilterOperationResult } from '@workbench/canvas-engine/engine';
import type * as WorkbenchUI from '@workbench/components/ui';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createControlLayer, createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { LayerFilterOperationButton } from './LayerFilterOperationButton';

interface CapturedButtonProps {
  children?: ReactNode;
  disabled?: boolean;
  onClick?: () => void;
}

interface CapturedTooltipProps {
  children?: ReactNode;
  content?: ReactNode;
  disabled?: boolean;
}

const { captured, notifyError, thumbnailSubscription } = vi.hoisted(() => ({
  captured: {
    button: null as CapturedButtonProps | null,
    tooltip: null as CapturedTooltipProps | null,
  },
  notifyError: vi.fn(),
  thumbnailSubscription: vi.fn(),
}));

vi.mock('@workbench/components/ui', async (importOriginal) => {
  const actual = await importOriginal<typeof WorkbenchUI>();

  return {
    ...actual,
    Button: (props: CapturedButtonProps) => {
      captured.button = props;
      return <button disabled={props.disabled}>{props.children}</button>;
    },
    Tooltip: (props: CapturedTooltipProps) => {
      captured.tooltip = props;
      return props.children;
    },
  };
});
vi.mock('@workbench/useNotify', () => ({
  useNotify: () => ({ error: notifyError, info: vi.fn(), success: vi.fn() }),
}));
vi.mock('@workbench/widgets/canvas/engineStoreHooks', () => ({
  useLayerThumbnailVersion: thumbnailSubscription,
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

type TestLayer = ReturnType<typeof createControlLayer> | ReturnType<typeof createEmptyPaintLayer>;

const createEngine = (
  hasExportableContent: boolean,
  launch: () => StartFilterOperationResult | undefined = () => 'started'
) => {
  const hasExportableLayerContent = vi.fn(() => hasExportableContent);
  const startFilterOperation = vi.fn(launch);
  const engine = { hasExportableLayerContent, startFilterOperation } as unknown as CanvasEngine;

  return { engine, hasExportableLayerContent, startFilterOperation };
};

const renderButton = (engine: CanvasEngine | null, layer: TestLayer, onOperationStarted = vi.fn()) => {
  renderToStaticMarkup(
    <ChakraProvider value={system}>
      <LayerFilterOperationButton engine={engine} layer={layer} onOperationStarted={onOperationStarted} />
    </ChakraProvider>
  );

  expect(captured.button).not.toBeNull();
  expect(captured.tooltip).not.toBeNull();

  return {
    button: captured.button!,
    onOperationStarted,
    tooltip: captured.tooltip!,
  };
};

beforeEach(() => {
  captured.button = null;
  captured.tooltip = null;
  notifyError.mockClear();
  thumbnailSubscription.mockClear();
});

describe('LayerFilterOperationButton eligibility', () => {
  it.each([
    {
      expectedReasonKey: 'widgets.layers.actions.notReady',
      getEngine: () => null,
      getLayer: () => createEmptyPaintLayer('Raster', 'raster'),
      scenario: 'missing engine',
    },
    {
      expectedReasonKey: 'widgets.layers.actions.disabled',
      getEngine: () => createEngine(true).engine,
      getLayer: () => ({ ...createEmptyPaintLayer('Raster', 'raster'), isEnabled: false }),
      scenario: 'disabled layer',
    },
    {
      expectedReasonKey: 'widgets.layers.actions.locked',
      getEngine: () => createEngine(true).engine,
      getLayer: () => ({ ...createEmptyPaintLayer('Raster', 'raster'), isLocked: true }),
      scenario: 'locked layer',
    },
    {
      expectedReasonKey: 'widgets.layers.actions.empty',
      getEngine: () => createEngine(false).engine,
      getLayer: () => createEmptyPaintLayer('Raster', 'raster'),
      scenario: 'empty layer',
    },
  ])('disables a $scenario with its stable tooltip reason', ({ expectedReasonKey, getEngine, getLayer }) => {
    const { button, tooltip } = renderButton(getEngine(), getLayer());

    expect(button.disabled).toBe(true);
    expect(tooltip.disabled).toBe(false);
    expect(tooltip.content).toBe(expectedReasonKey);
  });

  it.each([
    ['raster', createEmptyPaintLayer('Raster', 'raster')],
    ['control', createControlLayer('Control', 'control')],
  ] as const)('enables a filterable %s layer and subscribes to its thumbnail', (_type, layer) => {
    const { engine } = createEngine(true);
    const { button, tooltip } = renderButton(engine, layer);

    expect(button.children).toBe('widgets.layers.control.filter');
    expect(button.disabled).toBe(false);
    expect(tooltip.disabled).toBe(true);
    expect(tooltip.content).toBe('');
    expect(thumbnailSubscription).toHaveBeenCalledOnce();
    expect(thumbnailSubscription).toHaveBeenCalledWith(engine, layer.id);
  });
});

describe('LayerFilterOperationButton launch interaction', () => {
  it('starts synchronously and reports success exactly once', () => {
    const calls: string[] = [];
    const layer = createEmptyPaintLayer('Raster', 'raster');
    const { engine, startFilterOperation } = createEngine(true, () => {
      calls.push('engine');
      return 'started';
    });
    const onOperationStarted = vi.fn(() => calls.push('success'));
    const { button } = renderButton(engine, layer, onOperationStarted);

    button.onClick?.();

    expect(calls).toEqual(['engine', 'success']);
    expect(startFilterOperation).toHaveBeenCalledOnce();
    expect(startFilterOperation).toHaveBeenCalledWith(layer.id);
    expect(onOperationStarted).toHaveBeenCalledOnce();
    expect(notifyError).not.toHaveBeenCalled();
  });

  it.each([
    ['missing', 'widgets.layers.actions.missing'],
    ['disabled', 'widgets.layers.actions.disabled'],
    ['locked', 'widgets.layers.actions.locked'],
    ['unsupported', 'widgets.layers.actions.unsupported'],
    ['not-ready', 'widgets.layers.actions.notReady'],
  ] as const)('keeps properties open and reports an engine %s rejection', (result, reasonKey) => {
    const layer = createControlLayer('Control', 'control');
    const { engine, startFilterOperation } = createEngine(true, () => result);
    const onOperationStarted = vi.fn();
    const { button } = renderButton(engine, layer, onOperationStarted);

    button.onClick?.();

    expect(startFilterOperation).toHaveBeenCalledOnce();
    expect(startFilterOperation).toHaveBeenCalledWith(layer.id);
    expect(onOperationStarted).not.toHaveBeenCalled();
    expect(notifyError).toHaveBeenCalledOnce();
    expect(notifyError).toHaveBeenCalledWith('widgets.layers.actions.actionFailed', reasonKey);
  });

  it('keeps a missing-engine launch disabled and inert', () => {
    const layer = createEmptyPaintLayer('Raster', 'raster');
    const onOperationStarted = vi.fn();
    const { button } = renderButton(null, layer, onOperationStarted);

    button.onClick?.();

    expect(button.disabled).toBe(true);
    expect(onOperationStarted).not.toHaveBeenCalled();
    expect(notifyError).not.toHaveBeenCalled();
  });

  it('does not report an undefined engine launch result', () => {
    const layer = createEmptyPaintLayer('Raster', 'raster');
    const { engine, startFilterOperation } = createEngine(true, () => undefined);
    const onOperationStarted = vi.fn();
    const { button } = renderButton(engine, layer, onOperationStarted);

    button.onClick?.();

    expect(startFilterOperation).toHaveBeenCalledOnce();
    expect(onOperationStarted).not.toHaveBeenCalled();
    expect(notifyError).not.toHaveBeenCalled();
  });
});
