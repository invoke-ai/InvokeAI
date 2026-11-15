import type {
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasNodeContract,
  ShapeToolOptions,
} from '@workbench/canvas-engine/api';
import type { StructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import type { Project } from '@workbench/projectContracts';
import type { ToolFormProps } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { DEFAULT_SHAPE_OPTIONS, getDocumentLayer } from '@workbench/canvas-engine/api';
import { createStructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import { groupContract, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { PropertyGroup } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const harness = vi.hoisted(() => ({
  listeners: new Set<() => void>(),
  project: null as Project | null,
}));

vi.mock('@workbench/WorkbenchContext', async () => {
  const { useSyncExternalStore } = await import('react');
  const subscribe = (listener: () => void) => {
    harness.listeners.add(listener);
    return () => harness.listeners.delete(listener);
  };
  return {
    useActiveProjectSelector: (selector: (project: unknown) => unknown) =>
      selector(useSyncExternalStore(subscribe, () => harness.project)),
    useOptionalWorkbenchCommands: () => null,
    useWorkbenchCommands: () => ({ widgets: { patchValues: vi.fn() } }),
    useWorkbenchQueries: () => ({ getSnapshot: () => ({ activeProject: harness.project }) }),
  };
});
const notify = vi.hoisted(() => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => notify }));

import { shapeForm } from './ShapeOptions';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;

const i18n = createInstance();
beforeAll(async () => {
  const translation = (await (await fetch('/locales/en.json')).json()) as Record<string, unknown>;
  await i18n.init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const RECT_SOURCE: ShapeSource = {
  fill: '#ff0000',
  height: 40,
  kind: 'rect',
  stroke: '#0000ff',
  strokeWidth: 4,
  type: 'shape',
  width: 60,
};
const shapeLayer = (id: string): CanvasLayerContract =>
  ({ ...layerContract(id, 'raster', { name: 'Rect' }), source: RECT_SOURCE }) as CanvasLayerContract;

let stub: StructuralEngineStub;
let unsubscribe: () => void = () => undefined;
let shapeOptions: ShapeToolOptions;
const optionListeners = new Set<() => void>();
let host: HTMLDivElement | null = null;
let root: Root | null = null;

const publish = (): void => {
  harness.project = stub.project();
  harness.listeners.forEach((listener) => listener());
};

/** The form's engine: the structural stub over the real reducer and history, plus the tool's options store. */
const formEngine = (): ToolFormProps['engine'] =>
  ({
    ...stub.engine,
    interaction: {
      get: (key: string) => (key === 'shapeOptions' ? shapeOptions : undefined),
      set: (key: string, value: ShapeToolOptions) => {
        if (key === 'shapeOptions') {
          shapeOptions = value;
          optionListeners.forEach((listener) => listener());
        }
      },
      subscribe: (key: string, listener: () => void) => {
        if (key !== 'shapeOptions') {
          return () => undefined;
        }
        optionListeners.add(listener);
        return () => optionListeners.delete(listener);
      },
    },
    tools: { requestColorSample: () => Promise.resolve(null) },
  }) as unknown as ToolFormProps['engine'];

/** Renders the form's groups the way the Properties pane does. */
const mount = async (layers: readonly CanvasNodeContract[], selectedLayerId: string | null): Promise<void> => {
  unsubscribe();
  stub = createStructuralEngineStub({ layers, selectedLayerId });
  unsubscribe = stub.subscribe(publish);
  publish();
  shapeOptions = { ...DEFAULT_SHAPE_OPTIONS };
  const engine = formEngine();
  host = document.createElement('div');
  host.style.width = '300px';
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          {shapeForm.groups.map((group) => (
            <PropertyGroup key={group.id} id={group.id} label={i18n.t(group.labelKey)}>
              <group.body engine={engine} />
            </PropertyGroup>
          ))}
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const selectLayer = async (id: string | null): Promise<void> => {
  await act(() => {
    stub.ctx.dispatch({ id, type: 'setCanvasSelectedLayer' });
  });
};

const source = (id = 's1'): ShapeSource => {
  const layer = getDocumentLayer(stub.document(), id);
  if (layer?.type !== 'raster' || layer.source.type !== 'shape') {
    throw new Error(`${id} is not a shape layer`);
  }
  return layer.source;
};

const toolGroup = () => page.getByRole('group', { name: 'Shape' });
const appearanceGroup = () => page.getByRole('group', { name: 'Appearance' });
const kindGroup = () => toolGroup().getByRole('radiogroup', { name: 'Kind' });
/** The on/off button beside a color, not the color chip that shares its name. */
const clickToggle = (name: string) =>
  act(() =>
    appearanceGroup().element().querySelector<HTMLElement>(`button[aria-label="${name}"][aria-pressed]`)!.click()
  );
const hint = 'Select a paint layer or mask.';
const maskHint = 'Shapes add to the selected mask; fill and stroke set which parts, not their colors.';

afterEach(async () => {
  await act(() => root?.unmount());
  root = null;
  host?.remove();
  host = null;
  unsubscribe();
  unsubscribe = () => undefined;
  optionListeners.clear();
  vi.clearAllMocks();
});

describe('shape tool settings', () => {
  it('keeps every kind and Place on with a shape layer selected, and a kind only sets the next shape', async () => {
    await mount([shapeLayer('s1')], 's1');

    await expect.element(toolGroup().getByRole('radiogroup', { name: 'Place on' })).toBeVisible();
    for (const name of ['Rectangle', 'Ellipse', 'Triangle', 'Star', 'Polygon', 'Freehand']) {
      await expect.element(kindGroup().getByRole('radio', { exact: true, name })).toBeInTheDocument();
    }
    expect(kindGroup().getByRole('radio').elements()).toHaveLength(6);
    await expect.element(kindGroup().getByRole('radio', { name: 'Rectangle' })).toBeChecked();

    const polygon = kindGroup().getByRole('radio', { name: 'Polygon' });
    await userEvent.hover(polygon, { force: true });
    await expect.element(page.getByRole('tooltip')).toHaveTextContent('Polygon');
    await userEvent.click(polygon, { force: true });

    expect(shapeOptions.kind).toBe('polygon');
    await expect.element(polygon).toBeChecked();
    expect(source()).toBe(RECT_SOURCE);
    expect(stub.commits).toHaveLength(0);
  });

  it('picks a kind from the keyboard', async () => {
    await mount([layerContract('p1', 'raster')], 'p1');
    await act(() => (kindGroup().getByRole('radio', { name: 'Rectangle' }).element() as HTMLElement).focus());

    await userEvent.keyboard('{ArrowRight}');

    expect(shapeOptions.kind).toBe('ellipse');
    await expect.element(kindGroup().getByRole('radio', { name: 'Ellipse' })).toBeChecked();
  });

  it('shows the effective destination and restores the saved preference on an eligible selection', async () => {
    await mount([shapeLayer('s1'), layerContract('p1', 'raster'), layerContract('m1', 'inpaint_mask')], 's1');
    const target = toolGroup().getByRole('radiogroup', { name: 'Place on' });
    const selected = target.getByRole('radio', { name: 'Selected layer' });
    const newLayer = target.getByRole('radio', { name: 'New layer' });

    await expect.element(selected).toBeDisabled();
    await expect.element(newLayer).toBeChecked();
    await expect.element(toolGroup().getByText(hint)).toBeVisible();
    expect(target.element().getAttribute('aria-describedby')).toBe(toolGroup().getByText(hint).element().id);
    expect(shapeOptions.target).toBe('selected');
    await act(() => (newLayer.element() as HTMLElement).focus());
    await userEvent.keyboard('{ArrowLeft}');
    await expect.element(newLayer).toBeChecked();
    expect(shapeOptions.target).toBe('selected');

    await selectLayer('p1');
    await expect.element(selected).toBeEnabled();
    await expect.element(selected).toBeChecked();
    await expect.element(toolGroup().getByText(hint)).not.toBeInTheDocument();
    await userEvent.click(target.getByText('New layer'));
    expect(shapeOptions.target).toBe('new');
    await selectLayer('s1');
    await selectLayer('m1');
    await expect.element(newLayer).toBeChecked();
    await expect.element(toolGroup().getByText(maskHint)).not.toBeInTheDocument();
    await userEvent.click(target.getByText('Selected layer'));
    await expect.element(toolGroup().getByText(maskHint)).toBeVisible();
    expect(target.element().getAttribute('aria-describedby')).toBe(toolGroup().getByText(maskHint).element().id);
  });

  it.each([
    ['nothing', [], null],
    ['a group', [groupContract('g1')], 'g1'],
    ['a shape', [shapeLayer('s1')], 's1'],
    [
      'an image',
      [layerContract('i1', 'raster', { source: { image: { imageName: 'i', width: 1, height: 1 }, type: 'image' } })],
      'i1',
    ],
    [
      'a gradient',
      [
        layerContract('g1', 'raster', {
          source: {
            type: 'gradient',
            kind: 'linear',
            angle: 0,
            width: 10,
            height: 10,
            stops: [
              { color: '#000000', offset: 0 },
              { color: '#ffffff', offset: 1 },
            ],
          },
        }),
      ],
      'g1',
    ],
    [
      'text',
      [
        layerContract('t1', 'raster', {
          source: {
            type: 'text',
            content: 'Text',
            color: '#000000',
            fontFamily: 'sans-serif',
            fontSize: 12,
            fontWeight: 400,
            lineHeight: 1,
            align: 'left',
          },
        }),
      ],
      't1',
    ],
  ] as const)('disables Selected layer and explains New layer when %s is selected', async (_label, layers, id) => {
    await mount(layers, id);
    const target = toolGroup().getByRole('radiogroup', { name: 'Place on' });
    await expect.element(target.getByRole('radio', { name: 'Selected layer' })).toBeDisabled();
    await expect.element(target.getByRole('radio', { name: 'New layer' })).toBeChecked();
    await expect.element(toolGroup().getByText(hint)).toBeVisible();
    expect(shapeOptions.target).toBe('selected');
  });

  it.each([
    ['locked', { isLocked: true }],
    ['hidden', { isHidden: true }],
    ['loading', { source: { bitmap: { imageName: 'paint.png', width: 10, height: 10 }, type: 'paint' } }],
  ] as const)(
    'keeps the selected destination for a %s paint layer so the engine can report its refusal',
    async (_label, overrides) => {
      await mount([layerContract('p1', 'raster', overrides)], 'p1');
      const selected = toolGroup().getByRole('radio', { name: 'Selected layer' });
      await expect.element(selected).toBeEnabled();
      await expect.element(selected).toBeChecked();
      await expect.element(toolGroup().getByText(hint)).not.toBeInTheDocument();
    }
  );
});

describe('shape appearance', () => {
  it('edits only creation defaults even with a shape layer selected', async () => {
    await mount([shapeLayer('s1')], 's1');
    await expect.element(appearanceGroup().getByText('Defaults')).toBeVisible();
    await clickToggle('Fill');
    expect(shapeOptions.fillEnabled).toBe(false);
    await clickToggle('Stroke');
    expect(shapeOptions.strokeEnabled).toBe(true);
    const width = appearanceGroup().getByRole('slider', { name: 'Stroke width' });
    await act(async () => {
      (width.element() as HTMLElement).focus();
      await userEvent.keyboard('{ArrowRight>3/}');
    });
    expect(shapeOptions.strokeWidth).toBe(DEFAULT_SHAPE_OPTIONS.strokeWidth + 3);
    expect(source()).toBe(RECT_SOURCE);
    expect(stub.commits).toHaveLength(0);
  });

  it('edits the tool defaults with no shape layer selected', async () => {
    await mount([layerContract('p1', 'raster')], 'p1');
    await expect.element(appearanceGroup().getByText('Defaults')).toBeVisible();

    await clickToggle('Stroke');
    expect(shapeOptions.strokeEnabled).toBe(true);

    const width = appearanceGroup().getByRole('slider', { name: 'Stroke width' });
    await act(async () => {
      (width.element() as HTMLElement).focus();
      await userEvent.keyboard('{ArrowRight>2/}');
    });
    expect(shapeOptions.strokeWidth).toBe(DEFAULT_SHAPE_OPTIONS.strokeWidth + 2);
    expect(stub.commits).toHaveLength(0);
  });
});
