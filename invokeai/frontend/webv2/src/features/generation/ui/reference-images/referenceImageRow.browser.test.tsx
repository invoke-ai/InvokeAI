/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GenerationModelCatalogItem } from '@features/generation/contracts';
import type { GenerationIssue, GenerationIssueFix } from '@features/generation/core/generationIssues';
import type { GenerateModelConfig, GenerateReferenceImage, GenerateSettings } from '@features/generation/core/types';
import type { GenerateSettingsUpdate } from '@features/generation/ui/generateDebounce';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { type GenerationUiAdapter, GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { GenerateIssuesProvider } from '@features/generation/ui/shared/GenerateIssues';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, createRef, type Ref, useCallback, useImperativeHandle, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { commands, userEvent } from 'vitest/browser';

import { GenerateReferenceImagesContent } from './GenerateReferenceImagesSection';
declare module 'vitest/browser' {
  interface BrowserCommands {
    imeCompose: (steps: readonly { kind: 'commit' | 'compose'; text: string }[]) => Promise<void>;
  }
}

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

seedArchitectureCapabilities();

const SDXL: GenerateModelConfig = { base: 'sdxl', format: 'diffusers', key: 'sdxl', name: 'SDXL', type: 'main' };
const COGVIEW: GenerateModelConfig = { base: 'cogview4', key: 'cogview4', name: 'CogView4', type: 'main' };
const REDUX = { base: 'flux', key: 'redux', name: 'FLUX Redux', type: 'flux_redux' };
const NO_MODELS: [] = [];
const SDXL_ADAPTERS = [
  { base: 'sdxl', key: 'ip-adapter-sdxl', name: 'IP Adapter SDXL', type: 'ip_adapter' },
] as unknown as GenerationModelCatalogItem[];
const openManager = vi.fn();
const NO_VISION_COMPONENTS = {};
const undoable = vi.fn();
let catalog: readonly GenerationModelCatalogItem[] = NO_MODELS;
const ADAPTER = {
  gallery: { findImage: vi.fn() },
  models: {
    ModelSelect: ({ modelTypes, onChange }: { modelTypes: string[]; onChange: (model: unknown) => void }) => {
      return (
        <button type="button" onClick={() => onChange(modelTypes.includes('flux_redux') ? REDUX : null)}>
          Model picker
        </button>
      );
    },
    getBaseLabel: (base: string) => base,
    openManager,
    readCatalog: () => catalog,
  },
  notifications: { info: vi.fn(), undoable },
  project: { activeProjectId: 'project-1' },
  settings: {
    patchGenerateSettings: (patch: Partial<GenerateSettings>) => setStored(patch.referenceImages ?? stored),
    readGenerateValues: () => ({
      ...getDefaultGenerateSettings(SDXL),
      model: SDXL,
      modelKey: SDXL.key,
      referenceImages: stored,
    }),
  },
} as unknown as GenerationUiAdapter;

const reference = (id: string, name?: string): GenerateReferenceImage => ({
  config: {
    beginEndStepPct: [0.2, 0.8],
    clipVisionModel: 'ViT-H',
    image: { original: { image: { height: 768, image_name: `${id}.png`, width: 512 } } },
    method: 'composition',
    model: { base: 'sdxl', key: 'ip-adapter-sdxl', name: 'IP Adapter SDXL', type: 'ip_adapter' },
    type: 'ip_adapter',
    weight: 0.8,
  },
  id,
  isEnabled: true,
  ...(name === undefined ? {} : { name }),
});

let stored: GenerateReferenceImage[] = [];
const section = createRef<{ replace: (referenceImages: GenerateReferenceImage[]) => void }>();
const setStored = (referenceImages: GenerateReferenceImage[]) => section.current?.replace(referenceImages);

const needsAdapterIssues = (referenceImages: readonly GenerateReferenceImage[]): GenerationIssue[] =>
  referenceImages.flatMap((referenceImage, index) =>
    referenceImage.isEnabled && referenceImage.config.type === 'ip_adapter' && referenceImage.config.model === null
      ? [
          {
            fix: { kind: 'turnOffReferenceImages', referenceImageIds: [referenceImage.id] },
            id: `referenceImages:needsIpAdapter:${referenceImage.id}`,
            message: `${referenceImage.name ?? `Reference image ${index + 1}`} needs an IP Adapter model for SDXL.`,
            messageKey: 'widgets.generate.issues.referenceImageNeedsIpAdapter',
            messageValues: {
              index: index + 1,
              name: 'SDXL',
              ...(referenceImage.name === undefined ? {} : { referenceName: referenceImage.name }),
            },
            target: { field: 'referenceImage', referenceImageId: referenceImage.id, section: 'guidance' },
          } satisfies GenerationIssue,
        ]
      : []
  );

const Section = ({
  initial,
  model,
  models,
  ref,
}: {
  initial: GenerateReferenceImage[];
  model: GenerateModelConfig;
  models: readonly GenerationModelCatalogItem[];
  ref: Ref<{ replace: (referenceImages: GenerateReferenceImage[]) => void }>;
}) => {
  const [referenceImages, setReferenceImages] = useState(initial);
  const replace = useCallback((next: GenerateReferenceImage[]) => {
    stored = next;
    setReferenceImages(next);
  }, []);
  useImperativeHandle(ref, () => ({ replace }), [replace]);
  const applyFix = useCallback((fix: GenerationIssueFix) => {
    if (fix.kind === 'turnOffReferenceImages') {
      setReferenceImages((current) => {
        const next = current.map((referenceImage) =>
          fix.referenceImageIds.includes(referenceImage.id) ? { ...referenceImage, isEnabled: false } : referenceImage
        );
        stored = next;
        return next;
      });
    }
  }, []);
  const onCommit = useCallback(
    (update: GenerateSettingsUpdate) => {
      const next =
        typeof update === 'function'
          ? update({ referenceImages: stored } as GenerateSettings).referenceImages
          : (update.referenceImages ?? stored);
      replace(next);
    },
    [replace]
  );
  const onCommitImmediate = useCallback(
    (patch: Partial<GenerateSettings>) => (patch.referenceImages ? replace(patch.referenceImages) : undefined),
    [replace]
  );

  return (
    <GenerateIssuesProvider applyFix={applyFix} issues={needsAdapterIssues(referenceImages)}>
      <GenerateReferenceImagesContent
        models={models}
        referenceImages={referenceImages}
        selectedModel={model}
        visionComponents={NO_VISION_COMPONENTS}
        onCommit={onCommit}
        onCommitImmediate={onCommitImmediate}
      />
    </GenerateIssuesProvider>
  );
};

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (
  initial: GenerateReferenceImage[],
  model: GenerateModelConfig = SDXL,
  models: readonly GenerationModelCatalogItem[] = NO_MODELS
) => {
  stored = initial;
  catalog = models;
  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={ADAPTER}>
            <DndContext>
              <div data-hotkey-widget-type-id="generate" data-testid="section">
                <Section ref={section} initial={initial} model={model} models={models} />
              </div>
            </DndContext>
          </GenerationUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};
const row = (id: string) => host.querySelector<HTMLElement>(`[data-reference-image-id="${id}"]`)!;
const rowButton = (id: string) => row(id).querySelector<HTMLElement>('[data-list-primary]')!;
const nameOf = (id: string) => row(id).querySelector('[data-list-primary] [title]')?.textContent;
const renderedOrder = () =>
  [...host.querySelectorAll<HTMLElement>('[data-reference-image-id]')].map(
    (element) => element.dataset.referenceImageId
  );
const nameField = () => host.querySelector<HTMLInputElement>('input[aria-label="Rename"]');
const pressOn = async (element: HTMLElement, keys: string) => {
  await act(() => element.focus());
  await act(() => userEvent.keyboard(keys));
};

beforeEach(() => {
  undoable.mockClear();
  host = document.createElement('div');
  host.style.width = '360px';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  await userEvent.unhover(document.body);
});

describe('reference image rows', () => {
  it('opens and closes the settings from the chevron and from Right and Left on the row', async () => {
    await render([reference('a')]);
    const chevron = row('a').querySelector<HTMLButtonElement>('button[aria-label="Reference image 1 settings"]')!;

    expect(chevron.getAttribute('aria-expanded')).toBe('false');
    await act(() => chevron.click());
    expect(chevron.getAttribute('aria-expanded')).toBe('true');
    expect(document.getElementById(chevron.getAttribute('aria-controls')!)).not.toBeNull();
    expect(row('a').querySelector('[aria-label="Mode"]')).not.toBeNull();
    expect(row('a').textContent).toContain('ViT-H');
    expect(row('a').querySelector('button[aria-label="Crop Reference image 1"]')).not.toBeNull();
    expect(row('a').textContent).not.toContain('Advanced');

    await pressOn(rowButton('a'), '{ArrowLeft}');
    expect(chevron.getAttribute('aria-expanded')).toBe('false');
    expect(row('a').querySelector('[aria-label="Mode"]')).toBeNull();

    await pressOn(rowButton('a'), '{ArrowRight}');
    expect(chevron.getAttribute('aria-expanded')).toBe('true');
    expect(document.activeElement).toBe(rowButton('a'));
  });
});

describe('renaming a reference', () => {
  it('renames on a double-click, committing with Enter and handing focus back to the row', async () => {
    await render([reference('a'), reference('b')]);

    await act(() => userEvent.dblClick(rowButton('a')));
    const field = nameField()!;

    expect(field).not.toBeNull();
    expect(document.activeElement).toBe(field);
    expect([field.selectionStart, field.selectionEnd]).toEqual([0, 'Reference image 1'.length]);
    expect(field.closest('button')).toBeNull();

    await act(() => userEvent.keyboard('  Hero pose  {Enter}'));

    expect(nameField()).toBeNull();
    expect(nameOf('a')).toBe('Hero pose');
    expect(stored[0]?.name).toBe('Hero pose');
    await expect.poll(() => document.activeElement).toBe(rowButton('a'));
  });

  it('leaves Enter and Escape to an IME while it composes, and keeps Escape from reaching the panel', async () => {
    const panelKeys = vi.fn();
    document.body.addEventListener('keydown', panelKeys);
    await render([reference('a', 'Hero pose')]);

    await pressOn(rowButton('a'), '{F2}');
    await act(async () => {
      await commands.imeCompose([{ kind: 'compose', text: 'にほん' }]);
    });
    for (const key of ['Enter', 'Escape']) {
      await act(() =>
        nameField()!.dispatchEvent(
          new KeyboardEvent('keydown', { bubbles: true, cancelable: true, isComposing: true, key, keyCode: 229 })
        )
      );
    }
    expect(nameField()).not.toBeNull();

    await act(async () => {
      await commands.imeCompose([{ kind: 'commit', text: '日本' }]);
    });
    await act(() => userEvent.keyboard('{Escape}'));

    expect(nameField()).toBeNull();
    expect(stored[0]?.name).toBe('Hero pose');
    expect(panelKeys).not.toHaveBeenCalledWith(expect.objectContaining({ key: 'Escape' }));
    document.body.removeEventListener('keydown', panelKeys);

    await pressOn(rowButton('a'), '{F2}');
    await act(async () => {
      await commands.imeCompose([{ kind: 'compose', text: 'にほん' }]);
      await commands.imeCompose([{ kind: 'commit', text: '日本' }]);
    });
    await act(() => userEvent.keyboard('{Enter}'));
    expect(stored[0]?.name).toBe('日本');
  });

  it('keeps names with their images through a reorder and the removal of a sibling', async () => {
    await render([reference('a', 'Hero pose'), reference('b'), reference('c', 'Lighting')]);

    await pressOn(rowButton('a'), '{Alt>}{ArrowDown}{/Alt}');
    expect(renderedOrder()).toEqual(['b', 'a', 'c']);
    expect([nameOf('b'), nameOf('a'), nameOf('c')]).toEqual(['Reference image 1', 'Hero pose', 'Lighting']);

    await pressOn(rowButton('b'), '{Delete}');
    expect(renderedOrder()).toEqual(['a', 'c']);
    expect([nameOf('a'), nameOf('c')]).toEqual(['Hero pose', 'Lighting']);
    expect(stored.map(({ id, name }) => [id, name])).toEqual([
      ['a', 'Hero pose'],
      ['c', 'Lighting'],
    ]);
  });
});

describe('reordering and removing references', () => {
  it('moves with Alt+Up and Alt+Down, focus staying on the moved row, and stops at the ends', async () => {
    await render([reference('a'), reference('b'), reference('c')]);

    await pressOn(rowButton('c'), '{Alt>}{ArrowUp}{/Alt}');
    expect(renderedOrder()).toEqual(['a', 'c', 'b']);
    await expect.poll(() => document.activeElement).toBe(rowButton('c'));

    await act(() => userEvent.keyboard('{Alt>}{ArrowUp}{ArrowUp}{/Alt}'));
    expect(renderedOrder()).toEqual(['c', 'a', 'b']);
    await expect.poll(() => document.activeElement).toBe(rowButton('c'));
  });

  it('removes with Delete only, one row for a held key, and Undo puts it back in place as it was', async () => {
    const cropped = {
      ...reference('b', 'Palette'),
      config: {
        ...reference('b').config,
        image: {
          crop: {
            box: { height: 64, width: 64, x: 0, y: 0 },
            image: { height: 64, image_name: 'b-crop.png', width: 64 },
            ratio: null,
          },
          original: { image: { height: 768, image_name: 'b.png', width: 512 } },
        },
      },
    } as GenerateReferenceImage;
    await render([reference('a'), cropped, reference('c')]);

    await pressOn(rowButton('b'), '{Backspace}');
    expect(renderedOrder()).toEqual(['a', 'b', 'c']);

    await pressOn(rowButton('b'), '{Delete>4/}');
    expect(renderedOrder()).toEqual(['a', 'c']);
    expect(undoable).toHaveBeenCalledOnce();
    expect(undoable.mock.calls[0]![0]).toBe('Removed Palette');

    await act(() => undoable.mock.calls[0]![1].onUndo());
    expect(renderedOrder()).toEqual(['a', 'b', 'c']);
    expect(stored[1]).toEqual(cropped);
    await expect.poll(() => document.activeElement).toBe(rowButton('b'));
  });
});

describe('references the model cannot use', () => {
  it('stay listed but inert, badged, with Remove all above them', async () => {
    await render([reference('a', 'Hero pose')], COGVIEW);

    const button = rowButton('a');
    expect(row('a').textContent).toContain('Incompatible');
    expect(button.getAttribute('aria-disabled')).toBe('true');
    expect(row('a').querySelector('[inert]')).not.toBeNull();
    expect(host.textContent).toContain('Remove all');
    expect(host.querySelector('[data-reference-add]')).toBeNull();

    await act(() => button.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));
    await pressOn(button, '{F2}');
    await pressOn(button, '{Delete}');
    expect(nameField()).toBeNull();
    expect(stored).toHaveLength(1);
  });
});

describe('a reference that blocks Invoke', () => {
  const unadapted = (id: string): GenerateReferenceImage => {
    const referenceImage = reference(id);
    return referenceImage.config.type === 'ip_adapter'
      ? { ...referenceImage, config: { ...referenceImage.config, model: null } }
      : referenceImage;
  };
  const issueIcon = () => row('a').querySelector<HTMLButtonElement>('button[data-issue-ids]');
  const fixButton = () => document.querySelector<HTMLButtonElement>('[data-issue-id] [data-issue-fix]');

  it('turns off from the issue popover, which resolves the issue and hands focus to the switch', async () => {
    await render([unadapted('a')]);
    expect(row('a').querySelector('[data-issue-notes]')).toBeNull();

    await act(() => userEvent.click(issueIcon()!));
    await expect.poll(() => fixButton()?.textContent).toBe('Turn off');
    await act(() => userEvent.click(fixButton()!));

    expect(stored[0]?.isEnabled).toBe(false);
    expect(issueIcon()).toBeNull();
    await expect.poll(() => fixButton()).toBeNull();
    await expect.poll(() => document.activeElement).toBe(row('a').querySelector('[data-scope="switch"] input'));
  });
});

describe('a reference without its model', () => {
  const unadaptedNamed = (id: string, name?: string): GenerateReferenceImage => {
    const referenceImage = reference(id, name);
    return referenceImage.config.type === 'ip_adapter'
      ? { ...referenceImage, config: { ...referenceImage.config, model: null } }
      : referenceImage;
  };
  const chevron = () => row('a').querySelector<HTMLButtonElement>('button[aria-controls]')!;
  const issueIcon = () => row('a').querySelector<HTMLButtonElement>('button[data-issue-ids]');

  it('offers Choose model when a model switch leaves it without one, opening the settings on the picker', async () => {
    await render([reference('a')], SDXL, SDXL_ADAPTERS);
    expect(issueIcon()).toBeNull();

    await act(() => setStored([unadaptedNamed('a')]));
    expect(chevron().getAttribute('aria-expanded')).toBe('false');

    await act(() => userEvent.click(issueIcon()!));
    const action = () => document.querySelector<HTMLButtonElement>('[data-issue-id] [data-issue-action]');
    await expect.poll(() => action()?.textContent).toBe('Choose model');
    expect(document.querySelector('[data-issue-id] [data-issue-fix]')?.textContent).toBe('Turn off');
    await act(() => userEvent.click(action()!));

    expect(chevron().getAttribute('aria-expanded')).toBe('true');
    await expect.poll(() => document.activeElement?.closest('[data-reference-model]')).not.toBeNull();
    expect(stored[0]?.isEnabled).toBe(true);
  });
});
