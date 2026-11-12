import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createControlLayer } from '@workbench/canvas-engine/api';
import { layerContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { createDraftProject } from '@workbench/workbenchState';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { SelectionActions } from './SelectionOptionsRow';

interface CapturedButton {
  children?: ReactNode;
  /** The row disables through `aria-disabled` so the reason tooltip and keyboard focus keep working. */
  'aria-disabled'?: boolean;
}

const { activeProject, buttons, hasSelection, reasons, transformSession } = vi.hoisted(() => ({
  activeProject: { current: null as Project | null },
  transformSession: { current: null as { layerId: string } | null },
  buttons: new Map<string, CapturedButton>(),
  hasSelection: { current: true },
  /** Each action's disabled-reason tooltip, by its label. */
  reasons: new Map<string, ReactNode>(),
}));

vi.mock('@platform/ui/Tooltip', () => ({
  Tooltip: ({ children, content }: { children: { props: { children?: ReactNode } }; content: ReactNode }) => {
    reasons.set(String(children.props.children), content);
    return children;
  },
}));

vi.mock('@platform/ui/Button', () => ({
  Button: (props: CapturedButton) => {
    buttons.set(String(props.children), props);
    return <button aria-disabled={props['aria-disabled']}>{props.children}</button>;
  },
}));
vi.mock('@workbench/widgets/canvas/engineStoreHooks', () => ({
  useCanvasHasSelection: () => hasSelection.current,
  useTransformSession: () => transformSession.current,
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(activeProject.current!),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }) }));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { message?: string; name?: string; title?: string }) =>
      options?.title !== undefined
        ? `${options.title} / ${options.message}`
        : options?.name
          ? `${key}:${options.name}`
          : key,
  }),
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
      <SelectionActions engine={engine} />
    </ChakraProvider>
  );
  return new Map(buttons);
};

beforeEach(() => {
  buttons.clear();
  reasons.clear();
  transformSession.current = null;
  hasSelection.current = true;
});

describe('SelectionActions select all', () => {
  it('stays available with no selection and no eligible layer', () => {
    hasSelection.current = false;
    const captured = renderRow(createControlLayer({ base: 'sd-1', id: 'control-1', name: 'Control' }));
    expect(captured.get('widgets.canvas.toolOptions.selectAll')?.['aria-disabled']).toBe(false);
  });
});

describe('SelectionActions pixel target eligibility', () => {
  const rasterPaint = createEmptyPaintLayer('Raster', 'raster');
  const controlPaint = createControlLayer({ id: 'control', name: 'Control' });
  const rasterImage: CanvasLayerContract = {
    ...createEmptyPaintLayer('Raster image', 'raster-image'),
    source: { image: { height: 10, imageName: 'raster-image', width: 10 }, type: 'image' },
  };

  const mask = layerContract('mask', 'inpaint_mask', { name: 'Mask' });
  const shape: CanvasLayerContract = {
    ...rasterPaint,
    source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
  };

  it.each([
    { disabled: false, layer: rasterPaint, scenario: 'raster paint' },
    { disabled: false, layer: controlPaint, scenario: 'control paint' },
    { disabled: false, layer: rasterImage, scenario: 'raster image' },
    { disabled: false, layer: mask, scenario: 'inpaint mask' },
    { disabled: false, layer: layerContract('region', 'regional_guidance'), scenario: 'regional guidance' },
    { disabled: true, layer: { ...controlPaint, isLocked: true }, scenario: 'locked control' },
    { disabled: true, layer: { ...controlPaint, isEnabled: false }, scenario: 'disabled control' },
    { disabled: true, layer: shape, scenario: 'shape layer' },
  ])('sets Fill and Erase disabled=$disabled for $scenario', ({ disabled, layer }) => {
    const rendered = renderRow(layer);
    expect(rendered.get('widgets.canvas.toolOptions.fillSelection')?.['aria-disabled']).toBe(disabled);
    expect(rendered.get('widgets.canvas.toolOptions.eraseSelection')?.['aria-disabled']).toBe(disabled);
  });

  it('names the reason the brush and eraser would give', () => {
    renderRow({ ...mask, isHidden: true } as CanvasLayerContract);
    expect(reasons.get('widgets.canvas.toolOptions.fillSelection')).toBe(
      'widgets.canvas.editRefusal.displayHidden:Mask / widgets.canvas.editRefusal.displayHiddenMessage'
    );
    renderRow(shape);
    expect(reasons.get('widgets.canvas.toolOptions.eraseSelection')).toBe(
      'widgets.canvas.toolOptions.selectionNeedsPixelLayer'
    );
  });

  it('fills a transparency-locked layer but refuses to erase it', () => {
    const rendered = renderRow({ ...rasterPaint, isTransparencyLocked: true } as CanvasLayerContract);
    expect(rendered.get('widgets.canvas.toolOptions.fillSelection')?.['aria-disabled']).toBe(false);
    expect(rendered.get('widgets.canvas.toolOptions.eraseSelection')?.['aria-disabled']).toBe(true);
    expect(reasons.get('widgets.canvas.toolOptions.eraseSelection')).toBe(
      'widgets.canvas.editRefusal.alphaLocked:Raster / widgets.canvas.editRefusal.alphaLockedMessage'
    );
  });

  it('waits for an open transform session, naming its layer, as the engine does', () => {
    transformSession.current = { layerId: 'mask' };
    const rendered = renderRow(mask);
    for (const key of ['fillSelection', 'eraseSelection']) {
      expect(rendered.get(`widgets.canvas.toolOptions.${key}`)?.['aria-disabled']).toBe(true);
      expect(reasons.get(`widgets.canvas.toolOptions.${key}`)).toBe(
        'widgets.canvas.editRefusal.transformSession:Mask / widgets.canvas.editRefusal.transformSessionMessage'
      );
    }
  });

  it('keeps Lift to the layers it can lift from', () => {
    const rendered = renderRow(mask);
    expect(rendered.get('widgets.canvas.toolOptions.liftSelectionToLayer')?.['aria-disabled']).toBe(true);
  });

  it('disables every action with no live selection, even on an eligible layer', () => {
    hasSelection.current = false;
    const rendered = renderRow(rasterPaint);
    for (const key of ['fillSelection', 'eraseSelection', 'invertSelection', 'deselect']) {
      expect(rendered.get(`widgets.canvas.toolOptions.${key}`)?.['aria-disabled']).toBe(true);
    }
  });

  it('leaves invert and deselect enabled on an ineligible layer (they need only a selection)', () => {
    const rendered = renderRow(shape);
    expect(rendered.get('widgets.canvas.toolOptions.invertSelection')?.['aria-disabled']).toBe(false);
    expect(rendered.get('widgets.canvas.toolOptions.deselect')?.['aria-disabled']).toBe(false);
  });
});
