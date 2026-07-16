import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { CanvasStagingCandidateContract, Project, WidgetViewProps } from '@workbench/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { renderToStaticMarkup } from 'react-dom/server';
import { I18nextProvider } from 'react-i18next';
import { describe, expect, it, vi } from 'vitest';

const harness = vi.hoisted(() => ({ engine: null as CanvasEngine | null, project: null as Project | null }));

vi.mock('@dnd-kit/core', () => ({ useDndMonitor: () => undefined }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(harness.project!),
  useWorkbenchDispatch: () => () => undefined,
  useWorkbenchStore: () => ({
    getSnapshot: () => ({ activeProject: harness.project }),
    getState: () => ({ projects: harness.project ? [harness.project] : [] }),
  }),
}));
vi.mock('@workbench/useCanvasProjectMutationDispatch', () => ({
  useCanvasProjectMutationDispatch: () => () => undefined,
}));
vi.mock('@workbench/backend/progressImageStore', () => ({ useQueueItemProgressImage: () => null }));
vi.mock('./engineStoreHooks', () => ({ useCanvasOperation: () => null }));
vi.mock('./useCanvasEngine', () => ({ useCanvasEngine: () => harness.engine }));
vi.mock('./useCanvasGallerySave', () => ({
  useCanvasGallerySave: () => ({ isSaving: false, save: () => undefined }),
}));
vi.mock('./CanvasBottomControls', () => ({ CanvasBottomControls: () => null }));
vi.mock('./CanvasGlobalContextMenu', () => ({ CanvasGlobalContextMenu: () => null }));
vi.mock('./CanvasImageDropOverlay', () => ({ CanvasImageDropOverlay: () => null }));
vi.mock('./CanvasSaveToGallerySubmenu', () => ({ CanvasSaveToGallerySubmenu: () => null }));
vi.mock('./CanvasSurface', () => ({ CanvasSurface: () => null }));
vi.mock('./ToolStrip', () => ({ ToolStrip: () => null }));
vi.mock('@workbench/widgets/layers/LayerContextMenu', () => ({ CanvasLayerContextMenu: () => null }));

import { CanvasWidgetView } from './CanvasWidgetView';

const englishCatalogModules = import.meta.glob('../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const enCatalog = Object.values(englishCatalogModules)[0] as Record<string, unknown>;
const testI18n = createInstance();
await testI18n.init({
  initImmediate: false,
  lng: 'en',
  resources: { en: { translation: enCatalog } },
});

const candidate: CanvasStagingCandidateContract = {
  height: 40,
  imageName: 'ui-result.png',
  imageUrl: '/ui-result.png',
  placement: { height: 40, opacity: 1, width: 40, x: 0, y: 0 },
  queuedAt: '2026-07-16T00:00:00.000Z',
  sourceQueueItemId: 'queue-ui',
  thumbnailUrl: '/ui-result-thumb.png',
  width: 40,
};

const renderView = (): string =>
  renderToStaticMarkup(
    <ChakraProvider value={system}>
      <I18nextProvider i18n={testI18n}>
        <CanvasWidgetView {...({ runtime: { commands: {}, hotkeys: {} } } as unknown as WidgetViewProps)} />
      </I18nextProvider>
    </ChakraProvider>
  );

const getAcceptButtonTag = (markup: string): string => {
  const labelIndex = markup.indexOf('Accept to Layer');
  const buttonIndex = markup.lastIndexOf('<button', labelIndex);
  return markup.slice(buttonIndex, markup.indexOf('>', buttonIndex) + 1);
};

describe('CanvasWidgetView staged acceptance eligibility', () => {
  it('drives the rendered Chakra Accept button disabled state from interaction capabilities', () => {
    const store = createWorkbenchStore();
    const projectId = store.getState().activeProjectId;
    store.dispatch({ candidate, projectId, type: 'appendCanvasStagingCandidate' });
    harness.project = store.getState().projects[0]!;

    harness.engine = null;
    expect(getAcceptButtonTag(renderView())).toContain('disabled=""');

    harness.engine = { layers: { commitStagedImage: vi.fn() } } as unknown as CanvasEngine;
    expect(getAcceptButtonTag(renderView())).not.toContain('disabled=""');
  });
});
