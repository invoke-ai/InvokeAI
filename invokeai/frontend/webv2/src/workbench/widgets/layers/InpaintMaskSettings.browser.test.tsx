import type {
  CanvasInpaintMaskLayerContract,
  CanvasMaskFillContract,
  PreparedDocumentEdit,
} from '@workbench/canvas-engine/api';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { ComponentProps } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createDocumentModel } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createInstance } from 'i18next';
import { act, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it } from 'vitest';

import { InpaintMaskSettings } from './InpaintMaskSettings';

type Engine = ComponentProps<typeof InpaintMaskSettings>['engine'];

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const createLayer = (): CanvasInpaintMaskLayerContract =>
  ({
    id: 'mask-1',
    isEnabled: true,
    isLocked: false,
    mask: { bitmap: null, fill: { color: '#ff0000', style: 'solid' } },
    name: 'Inpaint Mask',
    opacity: 1,
    type: 'inpaint_mask',
  }) as unknown as CanvasInpaintMaskLayerContract;

const commits: { label: string; edit: PreparedDocumentEdit }[] = [];
let sampleRequests = 0;

const Harness = () => {
  const [layer, setLayer] = useState(createLayer);

  const engine = useMemo(() => {
    const apply = (mutation: CanvasProjectMutation): boolean => {
      const candidate = mutation as { type: string; config?: { mask?: { fill?: CanvasMaskFillContract } } };
      const fill = candidate.config?.mask?.fill;
      if (candidate.type === 'updateCanvasLayerConfig' && fill) {
        setLayer((current) => ({ ...current, mask: { ...current.mask, fill } }));
      }
      return true;
    };

    return {
      document: {
        model: () =>
          createDocumentModel(
            { ...createEmptyCanvasDocument(), selectedLayerId: layer.id, stacks: stacksFrom([layer]) },
            { editRevision: 0, projectId: 'test-project' }
          ),
      },
      layers: {
        beginStructuralPreview: () => ({
          apply,
          baseline: () => null,
          cancel: () => undefined,
          commit: (label: string, edit: PreparedDocumentEdit) => {
            commits.push({ edit, label });
            return { status: apply(edit.forward) ? ('committed' as const) : ('dispatch-rejected' as const) };
          },
          isActive: () => true,
        }),
        commitPrepared: (label: string, edit: PreparedDocumentEdit) => {
          commits.push({ edit, label });
          return { status: apply(edit.forward) ? ('committed' as const) : ('dispatch-rejected' as const) };
        },
        endStructuralPreview: () => undefined,
      },
      tools: {
        requestColorSample: () => {
          sampleRequests += 1;
          return Promise.resolve('#123456');
        },
      },
    } as unknown as Engine;
    // Rebuilt per layer change so the stub's model reflects the previewed layer, as the engine's does.
  }, [layer]);

  return <InpaintMaskSettings engine={engine} layer={layer} />;
};

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 30);
    });
  });

const render = async () => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

afterEach(async () => {
  commits.length = 0;
  sampleRequests = 0;
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const forwardFill = (edit: PreparedDocumentEdit): CanvasMaskFillContract =>
  (edit.forward as unknown as { config: { mask: { fill: CanvasMaskFillContract } } }).config.mask.fill;

describe('mask fill eyedropper', () => {
  it('samples the canvas through the engine and commits the picked fill color once', async () => {
    await render();

    const trigger = host!.querySelector<HTMLElement>('[aria-label="widgets.layers.maskFill.color"]')!;
    await settle(() => trigger.click());

    // With an engine present the picker offers the canvas sampler, not the screen eyedropper.
    const sampleButton = document.querySelector<HTMLElement>('[aria-label="common.colorPicker.sampleFromCanvas"]');
    expect(sampleButton).not.toBeNull();
    await settle(() => sampleButton!.click());

    expect(sampleRequests).toBe(1);
    expect(commits).toHaveLength(1);
    expect(forwardFill(commits[0]!.edit)).toEqual({ color: '#123456', style: 'solid' });
  });
});
