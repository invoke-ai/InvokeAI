/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { GenerationIssue } from '@features/generation/core/generationIssues';
import type { GenerateModelConfig, GenerateReferenceImage } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, MouseSensor, pointerWithin, useDraggable, useSensor, useSensors } from '@dnd-kit/core';
import { getGalleryItemDragData } from '@features/gallery/utility';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings, getMaxReferenceImages } from '@features/generation/core/baseGenerationPolicies';
import { type GenerationUiAdapter, GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { GenerateIssuesProvider } from '@features/generation/ui/shared/GenerateIssues';
import { dragWithMouse } from '@platform/browser/mouseDrag.testing';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { commands } from 'vitest/browser';

import { GenerateReferenceImagesContent } from './GenerateReferenceImagesSection';

declare module 'vitest/browser' {
  interface BrowserCommands {
    mouseDragPath: (points: readonly { x: number; y: number }[], frameWidth: number) => Promise<void>;
  }
}

vi.mock('@features/gallery', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  galleryImages: {
    resolveMany: (names: string[]) =>
      Promise.resolve(names.map((image_name) => ({ height: 512, image_name, width: 512 }))),
  },
}));

const i18n = i18next.createInstance();
// The shipped copy, so the counts read as a user sees them.
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

seedArchitectureCapabilities();

const SDXL: GenerateModelConfig = { base: 'sdxl', format: 'diffusers', key: 'sdxl', name: 'SDXL', type: 'main' };
/** Read inside each test: the capability table is seeded per test. */
const max = () => getMaxReferenceImages(SDXL);
const NO_MODELS: [] = [];
const reference = (id: string): GenerateReferenceImage => ({
  config: {
    beginEndStepPct: [0, 1],
    clipVisionModel: 'ViT-H',
    image: { original: { image: { height: 512, image_name: `${id}.png`, width: 512 } } },
    method: 'full',
    model: null,
    type: 'ip_adapter',
    weight: 1,
  },
  id,
  isEnabled: true,
});

const info = vi.fn();
const undoable = vi.fn();
const patchGenerateSettings = vi.fn();
let storedValues: Record<string, unknown> | null = null;
const ADAPTER = {
  gallery: { findImage: vi.fn() },
  models: { ModelSelect: () => null, getBaseLabel: (base: string) => base },
  notifications: { info, undoable },
  project: { activeProjectId: 'project-1' },
  settings: { patchGenerateSettings, readGenerateValues: () => storedValues },
} as unknown as GenerationUiAdapter;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** Stands in for a Gallery selection being dragged. */
const DraggedSelection = ({ count }: { count: number }) => {
  const { listeners, setNodeRef } = useDraggable({
    data: getGalleryItemDragData(
      Array.from({ length: count }, (_, index) => ({ kind: 'image', name: `${index}.png` }))
    ),
    id: 'selection',
  });

  return (
    <button ref={setNodeRef} type="button" {...listeners}>
      selection
    </button>
  );
};

const Dnd = ({ children }: { children: ReactNode }) => {
  const sensors = useSensors(useSensor(MouseSensor, { activationConstraint: { distance: 6 } }));

  return (
    <DndContext collisionDetection={pointerWithin} sensors={sensors}>
      {children}
    </DndContext>
  );
};

const render = async ({
  dragged = 0,
  issues = [],
  onCommit = vi.fn(),
  onCommitImmediate = vi.fn(),
  referenceImages,
  selectedModel = SDXL,
}: {
  dragged?: number;
  issues?: GenerationIssue[];
  onCommit?: (update: unknown) => void;
  onCommitImmediate?: (patch: unknown) => void;
  referenceImages: GenerateReferenceImage[];
  selectedModel?: GenerateModelConfig;
}) => {
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={ADAPTER}>
            <GenerateIssuesProvider applyFix={vi.fn()} issues={issues}>
              <Dnd>
                {dragged > 0 ? <DraggedSelection count={dragged} /> : null}
                <div data-testid="section">
                  <GenerateReferenceImagesContent
                    models={NO_MODELS}
                    referenceImages={referenceImages}
                    selectedModel={selectedModel}
                    onCommit={onCommit}
                    onCommitImmediate={onCommitImmediate}
                  />
                </div>
              </Dnd>
            </GenerateIssuesProvider>
          </GenerationUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const addZone = () =>
  [...host!.querySelectorAll<HTMLButtonElement>('button')].find((button) =>
    button.textContent?.includes('Add reference image')
  )!;
const selection = () =>
  [...host!.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === 'selection')!;
const references = (count: number) => Array.from({ length: count }, (_, index) => reference(`ref-${index}`));

beforeEach(() => {
  host = document.createElement('div');
  host.style.width = '360px';
  document.body.append(host);
  root = createRoot(host);
  info.mockClear();
  undoable.mockClear();
  patchGenerateSettings.mockClear();
  storedValues = null;
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('removing every reference', () => {
  const COGVIEW: GenerateModelConfig = { base: 'cogview4', key: 'cogview4', name: 'CogView4', type: 'main' };

  it('puts the removed references back first on Undo, turning off any past the current model limit', async () => {
    const onCommitImmediate = vi.fn();
    const removed = references(2);

    await render({ onCommitImmediate, referenceImages: removed, selectedModel: COGVIEW });
    await act(() =>
      [...host!.querySelectorAll<HTMLButtonElement>('button')]
        .find((button) => button.textContent === 'Remove all')!
        .click()
    );

    expect(onCommitImmediate).toHaveBeenCalledExactlyOnceWith({ referenceImages: [] });
    expect(undoable).toHaveBeenCalledOnce();

    // Since then the user switched to SDXL and added references of their own.
    const added = Array.from({ length: max() }, (_, index) => reference(`new-${index}`));
    storedValues = { ...getDefaultGenerateSettings(SDXL), model: SDXL, modelKey: SDXL.key, referenceImages: added };
    await act(() => undoable.mock.calls[0]![1].onUndo());

    const [patch, projectId] = patchGenerateSettings.mock.calls[0] as [
      { referenceImages: GenerateReferenceImage[] },
      string,
    ];
    expect(projectId).toBe('project-1');
    expect(patch.referenceImages.map((referenceImage) => referenceImage.id)).toEqual([
      ...removed.map((referenceImage) => referenceImage.id),
      ...added.map((referenceImage) => referenceImage.id),
    ]);
    expect(patch.referenceImages.filter((referenceImage) => referenceImage.isEnabled)).toHaveLength(max());
    expect(patch.referenceImages.slice(max()).every((referenceImage) => !referenceImage.isEnabled)).toBe(true);
  });
});

describe('reference image slots', () => {
  it('counts the slots in use on the add zone, and says why it is disabled once they are full', async () => {
    expect(max()).toBeGreaterThan(1);

    await render({ referenceImages: references(1) });
    expect(addZone().getAttribute('aria-disabled')).toBeNull();
    expect(addZone().textContent).toContain(`1 of ${max()} used`);

    await render({ referenceImages: references(max()) });
    expect(addZone().getAttribute('aria-disabled')).toBe('true');
    expect(addZone().textContent).toContain(`All ${max()} slots are in use. Remove one to add another.`);
  });

  it('adds what fits from a larger drop and says how many did not', async () => {
    const referenceImages = references(max() - 1);
    // The form applies a reference update to its settings at once, as this does.
    const onCommit = vi.fn((update: unknown) =>
      typeof update === 'function' ? update({ referenceImages }) : undefined
    );

    await render({ dragged: max() + 2, onCommit, referenceImages });
    await dragWithMouse(selection(), addZone());

    await vi.waitFor(() => expect(info).toHaveBeenCalledOnce());
    expect(info).toHaveBeenCalledWith(
      `Added 1 of ${max() + 2} images`,
      `This model takes up to ${max()} reference images. Remove one to add another.`
    );
    expect(onCommit).toHaveBeenCalledOnce();
  });

  it('ends a real mouse drag released over the full zone, so the drag never sticks to the pointer', async () => {
    const onCommit = vi.fn();
    const centre = (element: Element) => {
      const rect = element.getBoundingClientRect();
      return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
    };

    await render({ dragged: 1, onCommit, referenceImages: references(max()) });
    const from = centre(selection());
    const to = centre(addZone());
    // The browser's own hit-testing for every move and the release; a disabled button there would swallow the mouseup.
    await commands.mouseDragPath(
      [from, { x: from.x, y: from.y + 12 }, { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 }, to],
      window.innerWidth
    );

    await vi.waitFor(() =>
      expect(info).toHaveBeenCalledWith(
        `All ${max()} reference slots are in use`,
        'Remove a reference image to add another.'
      )
    );
    expect(onCommit).not.toHaveBeenCalled();
    expect(document.querySelector('[data-testid="section"]')?.textContent).not.toContain('Drop');
  });

  it('opens no picker from the full zone', async () => {
    await render({ referenceImages: references(max()) });
    await act(() => addZone().click());

    expect(document.querySelector('[data-scope="popover"][data-part="content"][data-state="open"]')).toBeNull();
  });

  it('takes a drop on a full set and says no slot is free, adding nothing', async () => {
    const onCommit = vi.fn();

    await render({ dragged: 1, onCommit, referenceImages: references(max()) });
    await dragWithMouse(selection(), addZone());

    await vi.waitFor(() =>
      expect(info).toHaveBeenCalledWith(
        `All ${max()} reference slots are in use`,
        'Remove a reference image to add another.'
      )
    );
    expect(onCommit).not.toHaveBeenCalled();
  });

  it("shows a reference's issue on its own card and the limit above the cards", async () => {
    const referenceImages = references(2);
    const cardIssue: GenerationIssue = {
      fix: { kind: 'turnOffReferenceImages', referenceImageIds: ['ref-1'] },
      id: 'referenceImages:needsIpAdapter:ref-1',
      message: 'Reference image 2 needs an IP Adapter model for SDXL.',
      target: { field: 'referenceImage', referenceImageId: 'ref-1', section: 'guidance' },
    };
    const limitIssue: GenerationIssue = {
      id: 'referenceImages:limit',
      message: 'SDXL supports at most 1 reference image. Turn off or remove the extra ones.',
      target: { field: 'referenceImages', section: 'guidance' },
    };

    await render({ issues: [cardIssue, limitIssue], referenceImages });

    const cards = [...host!.querySelectorAll<HTMLElement>('[data-issue-notes]')];
    // The nearest ancestor of the second card's title holding notes is that card.
    let secondCard: HTMLElement | null = [...host!.querySelectorAll('p')].find(
      (title) => title.textContent === 'Reference image 2'
    )!;
    while (secondCard && !secondCard.querySelector('[data-issue-notes]')) {
      secondCard = secondCard.parentElement;
    }

    expect(cards).toHaveLength(2);
    expect(secondCard?.textContent).toContain(cardIssue.message);
    expect(secondCard?.textContent).not.toContain(limitIssue.message);
    expect(secondCard?.querySelector('[data-issue-fix]')?.textContent).toBe('Turn off');
    expect(cards[0]?.textContent).toBe(limitIssue.message);
  });
});
