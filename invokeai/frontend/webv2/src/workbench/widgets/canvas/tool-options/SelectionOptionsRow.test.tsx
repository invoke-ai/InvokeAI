import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createControlLayer, createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { createDraftProject } from '@workbench/workbenchState';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { SelectionActions } from './SelectionOptionsRow';

interface CapturedButton {
  children?: ReactNode;
  /** The row disables through `aria-disabled` so the reason tooltip and keyboard focus keep working. */
  'aria-disabled'?: boolean;
}

const { activeProject, buttons, hasSelection } = vi.hoisted(() => ({
  activeProject: { current: null as Project | null },
  buttons: new Map<string, CapturedButton>(),
  hasSelection: { current: true },
}));

vi.mock('@platform/ui/Button', () => ({
  Button: (props: CapturedButton) => {
    buttons.set(String(props.children), props);
    return <button aria-disabled={props['aria-disabled']}>{props.children}</button>;
  },
}));
vi.mock('@workbench/widgets/canvas/engineStoreHooks', () => ({
  useCanvasHasSelection: () => hasSelection.current,
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(activeProject.current!),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }) }));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const engine = {
  selection: {
    deselect: vi.fn(),
    eraseSelection: vi.fn(),
    fillSelection: vi.fn(),
    invertSelection: vi.fn(),
    liftSelectionToLayer: vi.fn(),
    selectAll: vi.fn(),
  },
} as unknown as CanvasEngine;

const renderRow = (layer: CanvasLayerContract): Map<string, CapturedButton> => {
  const project = createDraftProject([]);
  project.canvas.document = {
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    stacks: stacksFrom([layer]),
    selectedLayerId: layer.id,
    version: 3,
    width: 100,
  };
  activeProject.current = project;
  renderToStaticMarkup(
    <ChakraProvider value={system}>
      <SelectionActions engine={engine} isSurfaceInteractionLocked={false} />
    </ChakraProvider>
  );
  return new Map(buttons);
};

beforeEach(() => {
  buttons.clear();
  hasSelection.current = true;
});

describe('SelectionActions select all', () => {
  it('stays available with no selection and no eligible layer', () => {
    hasSelection.current = false;
    const captured = renderRow(createControlLayer('Control', 'control-1', 'sd-1', null));
    expect(captured.get('widgets.canvas.toolOptions.selectAll')?.['aria-disabled']).toBe(false);
  });
});

describe('SelectionActions pixel target eligibility', () => {
  const rasterPaint = createEmptyPaintLayer('Raster', 'raster');
  const controlPaint = createControlLayer('Control', 'control');
  const rasterImage: CanvasLayerContract = {
    ...createEmptyPaintLayer('Raster image', 'raster-image'),
    source: { image: { height: 10, imageName: 'raster-image', width: 10 }, type: 'image' },
  };

  it.each([
    { disabled: false, layer: rasterPaint, scenario: 'raster paint' },
    { disabled: false, layer: controlPaint, scenario: 'control paint' },
    { disabled: true, layer: { ...controlPaint, isLocked: true }, scenario: 'locked control' },
    { disabled: true, layer: { ...controlPaint, isEnabled: false }, scenario: 'disabled control' },
    { disabled: true, layer: rasterImage, scenario: 'raster image' },
  ])('sets Fill and Erase disabled=$disabled for $scenario', ({ disabled, layer }) => {
    const rendered = renderRow(layer);
    expect(rendered.get('widgets.canvas.toolOptions.fillSelection')?.['aria-disabled']).toBe(disabled);
    expect(rendered.get('widgets.canvas.toolOptions.eraseSelection')?.['aria-disabled']).toBe(disabled);
  });

  it('disables every action with no live selection, even on an eligible layer', () => {
    hasSelection.current = false;
    const rendered = renderRow(rasterPaint);
    for (const key of ['fillSelection', 'eraseSelection', 'invertSelection', 'deselect']) {
      expect(rendered.get(`widgets.canvas.toolOptions.${key}`)?.['aria-disabled']).toBe(true);
    }
  });

  it('leaves invert and deselect enabled on an ineligible layer (they need only a selection)', () => {
    const rendered = renderRow(rasterImage);
    expect(rendered.get('widgets.canvas.toolOptions.invertSelection')?.['aria-disabled']).toBe(false);
    expect(rendered.get('widgets.canvas.toolOptions.deselect')?.['aria-disabled']).toBe(false);
  });
});
