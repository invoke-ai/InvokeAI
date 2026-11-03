import type { CanvasControlLayerContract, CanvasNodeContract } from '@workbench/canvas-engine/api';
import type { PreparedCommitOutcome } from '@workbench/widgets/canvas/useStructuralCommit';

import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createInstance } from 'i18next';
import { renderToStaticMarkup } from 'react-dom/server';
import { I18nextProvider } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { clearLayerPropertiesRequest, getLayerPropertiesRequest } from './layerPropertiesRequestStore';
import { useAddLayer } from './useAddLayer';

const harness = vi.hoisted(() => ({
  inserted: [] as unknown[],
  outcome: { status: 'committed' } as unknown,
}));

vi.mock('@workbench/canvas-operations/react', () => ({ useCanvasEngine: () => null }));
vi.mock('@workbench/widgets/canvas/useStructuralCommit', () => ({
  // Records what the add prepares and answers as the engine would, without one.
  usePreparedCommit:
    () =>
    (_label: string, prepare: (model: unknown) => unknown): unknown => {
      prepare({
        document: createEmptyCanvasDocument(),
        prepare: (request: { nodes: unknown[] }) => {
          harness.inserted.push(...request.nodes);
          return request;
        },
      });
      return harness.outcome;
    },
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => 'project-1',
  useActiveProjectSelector: (selector: (project: unknown) => unknown) =>
    selector({ canvas: { document: createEmptyCanvasDocument() } }),
}));
vi.mock('./useSelectedModelBase', () => ({ useSelectedModelBase: () => 'sdxl' }));

const i18n = createInstance();
void i18n.init({ initAsync: false, lng: 'en', resources: {} });

type AddLayer = ReturnType<typeof useAddLayer>;

const Probe = ({ onRender }: { onRender: (addLayer: AddLayer) => void }) => {
  onRender(useAddLayer());
  return null;
};

const rendered: AddLayer[] = [];
const recordRender = (addLayer: AddLayer): void => {
  rendered.push(addLayer);
};

const renderAddLayer = (): AddLayer => {
  renderToStaticMarkup(
    <I18nextProvider i18n={i18n}>
      <Probe onRender={recordRender} />
    </I18nextProvider>
  );
  return rendered.at(-1)!;
};

afterEach(() => {
  clearLayerPropertiesRequest();
  harness.inserted = [];
  harness.outcome = { status: 'committed' } as PreparedCommitOutcome;
});

describe('useAddLayer', () => {
  it('adds an untyped control layer and asks Properties to show it', () => {
    renderAddLayer()('control');

    const [layer] = harness.inserted as CanvasControlLayerContract[];
    expect(layer).toMatchObject({ adapter: { model: null }, type: 'control' });
    expect(layer).not.toHaveProperty('controlType');
    expect(getLayerPropertiesRequest()?.layerId).toBe(layer!.id);
  });

  it('leaves Properties alone when the insert is refused, and for other layer kinds', () => {
    harness.outcome = { status: 'unchanged' } satisfies PreparedCommitOutcome;
    const addLayer = renderAddLayer();

    addLayer('control');
    expect(getLayerPropertiesRequest()).toBeNull();

    harness.outcome = { status: 'committed' } as PreparedCommitOutcome;
    addLayer('raster');
    expect((harness.inserted.at(-1) as CanvasNodeContract).type).toBe('raster');
    expect(getLayerPropertiesRequest()).toBeNull();
  });
});
