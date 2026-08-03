/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Box, ChakraProvider } from '@chakra-ui/react';
import { DndContext, PointerSensor, useDraggable, useSensor, useSensors } from '@dnd-kit/core';
import { getGalleryItemDragData } from '@features/gallery/utility';
import { PositivePromptField } from '@features/generation/ui/promptFields/PositivePromptField';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en', resources: { en: { translation: {} } } });

const stubs = vi.hoisted(() => ({
  ModelSelect: () => null,
  selectedImage: { imageName: 'selected.png', imageUrl: '/i/selected.png', thumbnailUrl: '/t/selected.png' },
}));

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  // Stubbed alongside the hook: the real one reads the context from its own
  // unmocked module instance, which no provider in this harness fills.
  GenerationModelSelect: stubs.ModelSelect,
  useGenerationUi: () => ({
    gallery: { selectedImage: stubs.selectedImage },
    models: {
      catalog: [{ base: 'any', key: 'llava-1', name: 'LLaVA', type: 'llava_onevision' }],
      ensureLoaded: vi.fn(),
    },
    notifications: { reportError: vi.fn() },
    project: { activeProjectId: 'project-1' },
    promptHistory: { clear: vi.fn(), items: [] },
  }),
}));

vi.mock('@features/generation/data/wildcards', () => ({
  createWildcard: vi.fn(),
  deleteWildcard: vi.fn(),
  invalidateWildcardDependents: vi.fn(),
  updateWildcard: vi.fn(),
  wildcardsQueryOptions: () => ({ queryFn: () => Promise.resolve([]), queryKey: ['generation', 'wildcards'] }),
}));

const DROPPED_IMAGE_NAME = 'dropped.png';
const dragData = getGalleryItemDragData([{ kind: 'image', name: DROPPED_IMAGE_NAME }]);

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const GalleryDragSource = () => {
  const { listeners, setNodeRef } = useDraggable({ data: dragData, id: 'gallery-source' });

  return (
    <button
      ref={setNodeRef}
      {...listeners}
      data-testid="drag-source"
      style={{ height: 80, left: 20, position: 'fixed', top: 20, touchAction: 'none', width: 80 }}
      type="button"
    />
  );
};

const Harness = () => {
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

  return (
    <DndContext sensors={sensors}>
      <GalleryDragSource />
      <Box left="300px" position="fixed" top="20px" w="400px">
        <PositivePromptField
          heightPx={120}
          loras={[]}
          projectId="project-1"
          selectedModel={undefined}
          showSyntaxHighlighting={false}
          value=""
          onChange={vi.fn()}
          onResizeEnd={vi.fn()}
          onUsePrompt={vi.fn()}
        />
      </Box>
    </DndContext>
  );
};

const settle = (action: () => void, delay = 40): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, delay);
    });
  });

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};

const dropImageOnPromptBox = async ({ release = true } = {}) => {
  const source = host!.querySelector<HTMLElement>('[data-testid="drag-source"]')!;
  const box = host!.querySelector('textarea')!.getBoundingClientRect();
  const x = box.left + box.width / 2;
  const y = box.top + box.height / 2;

  await settle(() => pointer('pointerdown', source, 50, 50));
  await settle(() => pointer('pointermove', source.ownerDocument, 90, 50));
  await settle(() => pointer('pointermove', source.ownerDocument, x, y));

  if (release) {
    await settle(() => pointer('pointerup', source.ownerDocument, x, y));
  }
};

const imageToPromptButton = () =>
  host!.querySelector<HTMLButtonElement>('button[aria-label="widgets.generate.imageToPrompt"]')!;

const popoverImageName = (): string | null => {
  const image = document.querySelector<HTMLImageElement>('[data-scope="popover"][data-part="content"] img');

  return image?.getAttribute('alt') ?? null;
};

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await settle(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <QueryClientProvider client={new QueryClient()}>
          <ChakraProvider value={system}>
            <Harness />
          </ChakraProvider>
        </QueryClientProvider>
      </I18nextProvider>
    )
  );
});

afterEach(async () => {
  await settle(() => root?.unmount(), 0);
  host?.remove();
  host = null;
  root = null;
});

describe('dropping a gallery image on the positive prompt box', () => {
  it('offers the prompt box as a drop target only while an image is being dragged', async () => {
    expect(host!.textContent).not.toContain('widgets.generate.dropImageToPrompt');

    await dropImageOnPromptBox({ release: false });

    expect(host!.textContent).toContain('widgets.generate.dropImageToPrompt');

    await settle(() => pointer('pointerup', host!.ownerDocument, 0, 0));

    expect(host!.textContent).not.toContain('widgets.generate.dropImageToPrompt');
  });

  it('opens the image-to-prompt popover on the image that was dropped', async () => {
    await dropImageOnPromptBox();

    expect(popoverImageName()).toBe(DROPPED_IMAGE_NAME);
  });

  it('hands the popover back to the gallery selection once it is dismissed', async () => {
    await dropImageOnPromptBox();
    expect(popoverImageName()).toBe(DROPPED_IMAGE_NAME);

    await act(async () => {
      await userEvent.keyboard('{Escape}');
    });
    await settle(() => imageToPromptButton().click());

    expect(popoverImageName()).toBe(stubs.selectedImage.imageName);
  });
});
