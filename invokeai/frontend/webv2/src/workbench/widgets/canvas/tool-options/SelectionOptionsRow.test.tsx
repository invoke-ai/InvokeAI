import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createControlLayer, createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { createDraftProject } from '@workbench/workbenchState';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { SelectionOptionsRow } from './SelectionOptionsRow';

interface CapturedButton {
  children?: ReactNode;
  disabled?: boolean;
}

const { activeProject, buttons, hasSelection } = vi.hoisted(() => ({
  activeProject: { current: null as Project | null },
  buttons: new Map<string, CapturedButton>(),
  hasSelection: { current: true },
}));

vi.mock('@platform/ui', () => ({
  Button: (props: CapturedButton) => {
    buttons.set(String(props.children), props);
    return <button disabled={props.disabled}>{props.children}</button>;
  },
}));
vi.mock('@workbench/widgets/canvas/engineStoreHooks', () => ({
  useCanvasHasSelection: () => hasSelection.current,
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(activeProject.current!),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const engine = {
  selection: {
    deselect: vi.fn(),
    eraseSelection: vi.fn(),
    fillSelection: vi.fn(),
    invertSelection: vi.fn(),
  },
} as unknown as CanvasEngine;

const renderRow = (layer: CanvasLayerContract): Map<string, CapturedButton> => {
  const project = createDraftProject([]);
  project.canvas.document = {
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    layers: [layer],
    selectedLayerId: layer.id,
    version: 2,
    width: 100,
  };
  activeProject.current = project;
  renderToStaticMarkup(
    <ChakraProvider value={system}>
      <SelectionOptionsRow
        engine={engine}
        hintKey="widgets.canvas.toolOptions.lassoHint"
        mode="replace"
        onModeChange={vi.fn()}
      />
    </ChakraProvider>
  );
  return new Map(buttons);
};

beforeEach(() => {
  buttons.clear();
  hasSelection.current = true;
});

describe('SelectionOptionsRow pixel target eligibility', () => {
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
    expect(rendered.get('widgets.canvas.toolOptions.fillSelection')?.disabled).toBe(disabled);
    expect(rendered.get('widgets.canvas.toolOptions.eraseSelection')?.disabled).toBe(disabled);
  });

  it('disables every action with no live selection, even on an eligible layer', () => {
    hasSelection.current = false;
    const rendered = renderRow(rasterPaint);
    for (const key of ['fillSelection', 'eraseSelection', 'invertSelection', 'deselect']) {
      expect(rendered.get(`widgets.canvas.toolOptions.${key}`)?.disabled).toBe(true);
    }
  });

  it('leaves invert and deselect enabled on an ineligible layer (they need only a selection)', () => {
    const rendered = renderRow(rasterImage);
    expect(rendered.get('widgets.canvas.toolOptions.invertSelection')?.disabled).toBe(false);
    expect(rendered.get('widgets.canvas.toolOptions.deselect')?.disabled).toBe(false);
  });
});
