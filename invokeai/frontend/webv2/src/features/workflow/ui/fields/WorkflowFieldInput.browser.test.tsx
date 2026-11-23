/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { FieldInputTemplate } from '@features/workflow/contracts';

import { ChakraProvider, Field } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { GalleryUiProvider, type GalleryUiAdapter } from '@features/gallery/react';
import { getWorkflowFieldInvalidReason } from '@features/workflow/utility';
import { toaster } from '@platform/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance, type i18n as I18nInstance } from 'i18next';
import { act, cloneElement, startTransition, useCallback, useState } from 'react';
import { createPortal } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, server, userEvent } from 'vitest/browser';

import { WorkflowFieldInput, type WorkflowFieldInputProps } from './WorkflowFieldInput';

declare module 'vitest/browser' {
  interface BrowserCommands {
    imeCompose: (steps: readonly { kind: 'commit' | 'compose'; text: string }[]) => Promise<void>;
  }
}

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const uploadImageMock = vi.fn();
const uploadVideoMock = vi.fn();
const resolveItemMock = vi.fn();

const pickerState = vi.hoisted(() => ({ accept: null as readonly string[] | null, isOpen: false }));
const workflowApiMock = vi.hoisted(() => ({ apiFetch: vi.fn(), apiFetchJson: vi.fn() }));
const workflowCommandsMock = vi.hoisted(() => ({ editGraph: vi.fn() }));
const findInGalleryMock = vi.hoisted(() => vi.fn());

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  apiFetch: workflowApiMock.apiFetch,
  apiFetchJson: workflowApiMock.apiFetchJson,
}));

vi.mock('@features/gallery/picker', () => ({
  GalleryPickerPopover: ({
    accept,
    children,
    onPick,
  }: {
    accept: readonly string[];
    children: React.ReactElement<{ onClick?: () => void }>;
    onPick: (item: unknown) => void;
  }) => {
    pickerState.accept = accept;

    return (
      <>
        {cloneElement(children, { onClick: () => onPick(SELECTED_GALLERY_VIDEO) })}
        {/* The real picker's content is portaled out of the field, as here. */}
        {pickerState.isOpen ? createPortal(<input aria-label="Search the picker" />, document.body) : null}
      </>
    );
  },
}));

vi.mock('@features/gallery', () => ({
  formatGalleryVideoDuration: (seconds: number) => `${seconds}s`,
  galleryItems: { resolve: (...args: unknown[]) => resolveItemMock(...args) },
  isGalleryBoardAttachable: (boardId: string) => boardId !== 'none' && !boardId.startsWith('by_date:'),
  galleryTransfers: {
    upload: (...args: unknown[]) => uploadImageMock(...args),
    uploadVideo: (...args: unknown[]) => uploadVideoMock(...args),
  },
}));

const modelSelectState = vi.hoisted(() => ({
  props: null as null | {
    excludeKeys?: ReadonlySet<string>;
    filter?: (model: Record<string, unknown>) => boolean;
    modelTypes: readonly string[];
    onChange: (model: unknown) => void;
    placeholder?: string;
  },
}));

vi.mock('@features/models/react', () => ({
  ModelSelect: (props: NonNullable<typeof modelSelectState.props>) => {
    modelSelectState.props = props;

    return (
      <button type="button" onClick={() => props.onChange(ADDABLE_LORA)}>
        {props.placeholder}
      </button>
    );
  },
}));

const SD_LORA = {
  base: 'sd-1',
  default_settings: { weight: 0.4 },
  hash: 'hash-sd',
  key: 'sd-lora',
  name: 'Detail Tweaker',
  type: 'lora',
};
const SDXL_LORA = { base: 'sdxl', hash: 'hash-sdxl', key: 'sdxl-lora', name: 'Pixel Art XL', type: 'lora' };
// Carries a model default weight, so "adds at the model default" cannot pass by coincidence.
const ADDABLE_LORA = {
  base: 'sd-1',
  default_settings: { weight: 0.4 },
  hash: 'hash-add',
  key: 'add-lora',
  name: 'Add Me',
  type: 'lora',
};

const galleryValues: Record<string, unknown> = {};
const graphNodes: unknown[] = [];
const graphEdges: unknown[] = [];
const workflowValues: Record<string, unknown> = {};
const projectSnapshot = {
  galleryValues,
  id: 'project-1',
  projectGraph: { edges: graphEdges, nodes: graphNodes },
  workflowValues,
};

vi.mock('@features/workflow/ui/WorkflowUiContext', () => ({
  useWorkflowProjectSelector: (selector: (project: typeof projectSnapshot) => unknown) => selector(projectSnapshot),
  useWorkflowUi: () => ({
    commands: workflowCommandsMock,
    findInGallery: findInGalleryMock,
    project: { getSnapshot: () => projectSnapshot },
  }),
}));

const VIDEO_TEMPLATE = {
  name: 'video',
  title: 'Video',
  type: { name: 'VideoField' },
} as unknown as FieldInputTemplate;

const SEED_TEMPLATE = {
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  input: 'any',
  maximum: 4_294_967_295,
  minimum: 0,
  multipleOf: null,
  name: 'seed',
  title: 'Seed',
  type: { cardinality: 'SINGLE', name: 'IntegerField' },
} as unknown as FieldInputTemplate;

const FRAME_INDEX_TEMPLATE = {
  name: 'frame_index',
  title: 'Frame Index',
  type: { cardinality: 'SINGLE', name: 'IntegerField' },
  uiComponent: 'video-frame-index',
} as unknown as FieldInputTemplate;

const SAVED_WORKFLOW_TEMPLATE = {
  input: 'direct',
  name: 'workflow_id',
  title: 'Workflow',
  type: { batch: false, cardinality: 'SINGLE', name: 'SavedWorkflowField' },
} as unknown as FieldInputTemplate;

const NUMERIC_ENUM_TEMPLATE = {
  name: 'max_seq_len',
  options: [256, 512],
  title: 'Maximum sequence length',
  type: { batch: false, cardinality: 'SINGLE', name: 'EnumField' },
} as unknown as FieldInputTemplate;

const makeFrameNode = (videoValue: { video_name: string } | undefined) => ({
  data: {
    inputs: {
      frame_index: { label: '', name: 'frame_index', value: -1 },
      video: { label: '', name: 'video', value: videoValue },
    },
    type: 'video_frame_extract',
  },
  id: 'frame-node',
  type: 'invocation',
});

const LORA_COLLECTION_TEMPLATE = {
  name: 'loras',
  title: 'LoRAs',
  type: { cardinality: 'SINGLE_OR_COLLECTION', name: 'LoRAField' },
  uiModelBase: ['sd-1', 'sd-2'],
  uiModelType: ['lora'],
} as unknown as FieldInputTemplate;

const loraEntry = (overrides: Record<string, unknown> = {}) => ({
  lora: { base: 'sd-1', hash: 'hash-sd', key: 'sd-lora', name: 'Detail Tweaker', type: 'lora' },
  weight: 0.75,
  ...overrides,
});

const SELECTED_GALLERY_VIDEO = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-01-01T00:00:00Z',
  durationSeconds: 5,
  fullUrl: '/api/v1/videos/i/clip.mp4/full',
  height: 480,
  isIntermediate: false,
  kind: 'video',
  name: 'clip.mp4',
  starred: false,
  thumbnailUrl: '/api/v1/videos/i/clip.mp4/thumbnail',
  width: 640,
};

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  uploadImageMock.mockReset();
  uploadVideoMock.mockReset();
  resolveItemMock.mockReset();
  resolveItemMock.mockResolvedValue(SELECTED_GALLERY_VIDEO);
  workflowApiMock.apiFetch.mockReset().mockResolvedValue(new Response());
  workflowApiMock.apiFetchJson.mockReset();
  workflowCommandsMock.editGraph.mockReset();
  // Module-level capture: without this the next test's wait is satisfied by the previous test's props.
  modelSelectState.props = null;
  queryClient.clear();
  delete galleryValues.selectedImage;
  graphNodes.length = 0;
  graphEdges.length = 0;
  delete workflowValues.batchCount;
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

// No retry backoff: failure states must be observable inside a test timeout.
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

const renderField = async (
  template: FieldInputTemplate,
  value: unknown,
  onChange: (value: unknown) => void,
  nodeId?: string,
  seedProps: Pick<WorkflowFieldInputProps, 'onSeedModeChange' | 'seedMode'> = {},
  i18n?: I18nInstance
) => {
  const field = (
    <ChakraProvider value={system}>
      <QueryClientProvider client={queryClient}>
        <DndContext>
          <WorkflowFieldInput nodeId={nodeId} template={template} value={value} onChange={onChange} {...seedProps} />
        </DndContext>
      </QueryClientProvider>
    </ChakraProvider>
  );
  await act(() => {
    root.render(i18n ? <I18nextProvider i18n={i18n}>{field}</I18nextProvider> : field);
  });
};

/** Real keystrokes: the weight regressions live between inputs, where a direct DOM write never goes. */
const typeWeight = async (input: HTMLInputElement, keys: string) => {
  await act(async () => {
    await userEvent.click(input);
    await userEvent.keyboard(`{Control>}a{/Control}${keys}`);
  });
};

const blurWeight = async (input: HTMLInputElement) => {
  await act(() => {
    input.blur();
  });
};

const findButton = (label: string): HTMLButtonElement => {
  const button = Array.from(host.querySelectorAll('button')).find((el) => el.textContent?.includes(label));

  if (!button) {
    throw new Error(`Button "${label}" not found`);
  }

  return button;
};

const BOOLEAN_TEMPLATE = {
  name: 'enabled',
  title: 'Enabled',
  type: { batch: false, cardinality: 'SINGLE', name: 'BooleanField' },
} as unknown as FieldInputTemplate;

describe('WorkflowFieldInput boolean', () => {
  it('toggles from a click on the switch when hosted in a Field with an external id', async () => {
    const onChange = vi.fn();

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Field.Root>
            <WorkflowFieldInput id="node-enabled-value" template={BOOLEAN_TEMPLATE} value={false} onChange={onChange} />
          </Field.Root>
        </ChakraProvider>
      );
    });

    const control = host.querySelector('[data-scope="switch"][data-part="control"]');

    if (!(control instanceof HTMLElement)) {
      throw new TypeError('Switch control not rendered');
    }

    await act(async () => {
      await userEvent.click(control);
    });

    expect(onChange).toHaveBeenCalledWith(true);
  });
});

describe('WorkflowFieldInput authored form controls', () => {
  const renderWithSettings = async (
    template: FieldInputTemplate,
    value: unknown,
    onChange: (value: unknown) => void,
    formSettings: WorkflowFieldInputProps['formSettings']
  ) => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <DndContext>
              <WorkflowFieldInput formSettings={formSettings} template={template} value={value} onChange={onChange} />
            </DndContext>
          </QueryClientProvider>
        </ChakraProvider>
      );
    });
  };

  it('restricts a dropdown string field to the authored options and commits the option value', async () => {
    const onChange = vi.fn();
    await renderWithSettings(fullTemplate('StringField'), 'slow', onChange, {
      component: 'dropdown',
      kind: 'string',
      options: [
        { label: 'Fast', value: 'fast' },
        { label: '', value: 'slow' },
      ],
    });

    expect(host.querySelector('input:not([type="hidden"]), textarea')).toBeNull();
    const trigger = host.querySelector<HTMLElement>('[data-scope="select"][data-part="trigger"]')!;
    expect(trigger.textContent).toContain('slow');

    await act(() => userEvent.click(trigger));
    const items = Array.from(document.querySelectorAll<HTMLElement>('[data-scope="select"][data-part="item"]'));
    expect(items.map((item) => item.textContent)).toEqual(['Fast', 'slow']);

    await act(() => userEvent.click(items[0]!));
    expect(onChange).toHaveBeenCalledWith('fast');
  });

  it('shows a slider for a bounded number field and keeps the input when an end is open', async () => {
    const bounded = fullTemplate('IntegerField', { maximum: 20, minimum: 4, multipleOf: 4 });
    await renderWithSettings(bounded, 8, vi.fn(), { component: 'slider', kind: 'number' });

    const slider = host.querySelector<HTMLElement>('[role="slider"]')!;
    expect(slider.getAttribute('aria-valuemin')).toBe('4');
    expect(slider.getAttribute('aria-valuemax')).toBe('20');
    expect(slider.getAttribute('aria-valuenow')).toBe('8');
    expect(host.querySelector('input[type="number"]')).toBeNull();

    await renderWithSettings(bounded, 8, vi.fn(), { component: 'number-input-and-slider', kind: 'number' });
    expect(host.querySelector('[role="slider"]')).not.toBeNull();
    expect(host.querySelector<HTMLInputElement>('[data-scope="number-input"] input')?.value).toBe('8');

    await renderWithSettings(fullTemplate('IntegerField', { minimum: 0 }), 8, vi.fn(), {
      component: 'slider',
      kind: 'number',
    });
    expect(host.querySelector('[role="slider"]')).toBeNull();
    expect(host.querySelector<HTMLInputElement>('input[type="number"]')?.value).toBe('8');
  });
});

describe('WorkflowFieldInput typed enums', () => {
  it.each([
    { label: 'numeric', options: [256, 512], selected: 512, next: 256 },
    { label: 'boolean', options: [false, true], selected: false, next: true },
    { label: 'string', options: ['small', 'large'], selected: 'large', next: 'small' },
  ])('shows the selected $label value and preserves its type when changed', async ({ options, selected, next }) => {
    const onChange = vi.fn();
    await renderField({ ...NUMERIC_ENUM_TEMPLATE, options }, selected, onChange);

    const trigger = host.querySelector<HTMLElement>('[data-scope="select"][data-part="trigger"]');
    expect(trigger?.textContent).toContain(String(selected));

    if (!trigger) {
      throw new Error('Numeric enum trigger not rendered');
    }
    await act(() => userEvent.click(trigger));

    const option = Array.from(document.querySelectorAll<HTMLElement>('[data-scope="select"][data-part="item"]')).find(
      (item) => item.textContent?.includes(String(next))
    );
    if (!option) {
      throw new Error('Numeric enum option not rendered');
    }
    await act(() => userEvent.click(option));
    expect(onChange).toHaveBeenCalledWith(next);
  });
});

describe('WorkflowFieldInput saved workflows', () => {
  it('retries a failed child when its already-selected workflow is picked again', async () => {
    const selectedWorkflow = {
      category: 'user',
      call_saved_workflow_compatibility: { is_callable: true, message: null, reason: 'ok' },
      description: '',
      is_public: false,
      name: 'Selected child',
      workflow_id: 'selected-child',
    };
    workflowApiMock.apiFetchJson.mockImplementation((path: string) =>
      path.includes('/i/')
        ? Promise.reject(new Error('Temporary detail failure'))
        : Promise.resolve({ items: [selectedWorkflow], page: 0, pages: 1, total: 1 })
    );
    const onChange = vi.fn();
    await renderField(SAVED_WORKFLOW_TEMPLATE, 'selected-child', onChange, 'call-node');
    await vi.waitFor(() =>
      expect(host.querySelector<HTMLInputElement>('input[role="combobox"]')?.value).toBe('Selected child')
    );
    await vi.waitFor(() =>
      expect(host.querySelector<HTMLButtonElement>('button[aria-label="common.retry"]')).not.toBeNull()
    );

    const input = host.querySelector<HTMLInputElement>('input[role="combobox"]');
    if (!input) {
      throw new Error('Saved workflow combobox not rendered');
    }
    await act(() => userEvent.click(input));
    const selectedItem = Array.from(
      document.querySelectorAll<HTMLElement>('[data-scope="combobox"][data-part="item"]')
    ).find((item) => item.textContent?.includes('Selected child'));
    if (!selectedItem) {
      throw new Error('Selected workflow option not rendered');
    }
    await act(() => userEvent.click(selectedItem));

    expect(workflowCommandsMock.editGraph).toHaveBeenCalledWith({
      nodeId: 'call-node',
      type: 'retryCallSavedWorkflow',
    });
    expect(workflowCommandsMock.editGraph).toHaveBeenCalledTimes(1);
    expect(onChange).not.toHaveBeenCalled();

    workflowCommandsMock.editGraph.mockClear();
    const retryButton = host.querySelector<HTMLButtonElement>('button[aria-label="common.retry"]');
    if (!retryButton) {
      throw new Error('Saved workflow retry button not rendered');
    }
    await act(() => userEvent.click(retryButton));
    expect(workflowCommandsMock.editGraph).toHaveBeenCalledOnce();

    workflowCommandsMock.editGraph.mockClear();
    await act(() => userEvent.click(input));
    await act(() => userEvent.keyboard('{ArrowDown}{Enter}'));
    expect(workflowCommandsMock.editGraph).toHaveBeenCalledOnce();
  });

  it('displays dynamic workflow names and marks incompatible workflows disabled', async () => {
    const onChange = vi.fn();
    workflowApiMock.apiFetchJson.mockImplementation((path: string) => {
      if (path.includes('is_public=true')) {
        return Promise.resolve({
          items: [
            {
              category: 'user',
              call_saved_workflow_compatibility: { is_callable: true, message: null, reason: 'ok' },
              description: '',
              is_public: true,
              name: 'Shared Dynamic Workflow',
              workflow_id: 'shared-workflow',
            },
          ],
          page: 0,
          pages: 1,
          total: 1,
        });
      }

      return Promise.resolve({
        items: [
          {
            category: 'default',
            call_saved_workflow_compatibility: {
              is_callable: false,
              message: 'Missing workflow return node',
              reason: 'missing_workflow_return',
            },
            description: '',
            is_public: true,
            name: 'Unsupported Dynamic Workflow',
            workflow_id: 'unsupported-workflow',
          },
        ],
        page: 0,
        pages: 1,
        total: 1,
      });
    });

    await renderField(SAVED_WORKFLOW_TEMPLATE, '', onChange);
    await vi.waitFor(() => expect(workflowApiMock.apiFetchJson).toHaveBeenCalledTimes(2));
    await vi.waitFor(() =>
      expect(host.querySelector<HTMLInputElement>('input[role="combobox"]')?.placeholder).toBe(
        'nodes.savedWorkflowSearch'
      )
    );
    const input = host.querySelector<HTMLInputElement>('input[role="combobox"]');

    if (!input) {
      throw new Error('Saved workflow combobox not rendered');
    }

    await act(async () => {
      await userEvent.click(input);
    });

    await vi.waitFor(() => {
      expect(document.body.textContent).toContain('Shared Dynamic Workflow');
      expect(document.body.textContent).toContain('Unsupported Dynamic Workflow');
    });

    const unsupportedItem = Array.from(document.querySelectorAll('[data-scope="combobox"][data-part="item"]')).find(
      (item) => item.textContent?.includes('Unsupported Dynamic Workflow')
    );

    expect(unsupportedItem?.hasAttribute('data-disabled')).toBe(true);

    const sharedItem = Array.from(document.querySelectorAll('[data-scope="combobox"][data-part="item"]')).find((item) =>
      item.textContent?.includes('Shared Dynamic Workflow')
    );

    if (!(sharedItem instanceof HTMLElement)) {
      throw new TypeError('Shared workflow option not rendered');
    }

    await act(async () => {
      await userEvent.click(sharedItem);
    });

    expect(onChange).toHaveBeenCalledWith('shared-workflow');
  });
});

describe('WorkflowFieldInput media inputs', () => {
  it('adopts a video picked from the gallery, shows its details, and clears it', async () => {
    const onChange = vi.fn();

    await renderField(VIDEO_TEMPLATE, undefined, onChange);
    await act(() => findButton('widgets.gallery.picker.chooseVideo').click());
    expect(onChange).toHaveBeenCalledWith({ video_name: 'clip.mp4' });

    await renderField(VIDEO_TEMPLATE, { video_name: 'clip.mp4' }, onChange);
    expect(host.querySelector('img')?.src).toContain('/api/v1/videos/i/clip.mp4/thumbnail');
    await vi.waitFor(() => {
      expect(host.textContent).toContain('640x480 · 5s');
    });

    await act(() => findButton('Clear').click());
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it('edits image collections as a list: picks append, remove drops one, clear empties', async () => {
    const onChange = vi.fn();
    const collectionTemplate = {
      name: 'images',
      title: 'Images',
      type: { cardinality: 'COLLECTION', name: 'ImageField' },
    } as unknown as FieldInputTemplate;

    await renderField(collectionTemplate, [{ image_name: 'a.png' }, { image_name: 'b.png' }], onChange);

    expect(host.textContent).not.toContain('nodes.inputModes.connection');
    expect(host.querySelectorAll('img')).toHaveLength(2);
    expect(host.querySelector<HTMLInputElement>('input[type="file"]')?.multiple).toBe(true);

    await act(() => findButton('common.add').click());
    expect(onChange).toHaveBeenLastCalledWith([
      { image_name: 'a.png' },
      { image_name: 'b.png' },
      { image_name: SELECTED_GALLERY_VIDEO.name },
    ]);

    const findButtons = [...host.querySelectorAll<HTMLButtonElement>('button[aria-label*="InGallery"]')];
    expect(findButtons).toHaveLength(2);
    await act(() => findButtons[1]!.click());
    expect(findInGalleryMock).toHaveBeenLastCalledWith({ kind: 'image', name: 'b.png' });

    await act(() => host.querySelector<HTMLButtonElement>('button[aria-label="Remove a.png"]')!.click());
    expect(onChange).toHaveBeenLastCalledWith([{ image_name: 'b.png' }]);

    await act(() => findButton('Clear').click());
    expect(onChange).toHaveBeenLastCalledWith([]);
  });

  it('adopts the files that uploaded when one of a multi-file batch fails', async () => {
    uploadImageMock.mockResolvedValueOnce({ imageName: 'first.png' }).mockRejectedValueOnce(new Error('413'));
    const createToast = vi.spyOn(toaster, 'create').mockImplementation(() => '');
    const onChange = vi.fn();
    const collectionTemplate = {
      name: 'images',
      title: 'Images',
      type: { cardinality: 'COLLECTION', name: 'ImageField' },
    } as unknown as FieldInputTemplate;

    await renderField(collectionTemplate, [{ image_name: 'a.png' }], onChange);

    const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    const transfer = new DataTransfer();

    transfer.items.add(new File(['1'], 'one.png', { type: 'image/png' }));
    transfer.items.add(new File(['2'], 'two.png', { type: 'image/png' }));
    fileInput.files = transfer.files;
    await act(() => fileInput.dispatchEvent(new Event('change', { bubbles: true })));

    await vi.waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([{ image_name: 'a.png' }, { image_name: 'first.png' }]);
    });
    await vi.waitFor(() => {
      expect(createToast).toHaveBeenCalledWith(expect.objectContaining({ description: 'two.png', type: 'error' }));
    });
    createToast.mockRestore();
  });

  it('uploads a picked file and adopts the uploaded video', async () => {
    uploadVideoMock.mockResolvedValue({ kind: 'video', name: 'uploaded.mp4' });
    const onChange = vi.fn();

    await renderField(VIDEO_TEMPLATE, undefined, onChange);
    await act(() => findButton('Upload').click());

    const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    const transfer = new DataTransfer();

    transfer.items.add(new File(['data'], 'clip.mp4', { type: 'video/mp4' }));
    fileInput.files = transfer.files;
    await act(() => fileInput.dispatchEvent(new Event('change', { bubbles: true })));

    await vi.waitFor(() => {
      expect(onChange).toHaveBeenCalledWith({ video_name: 'uploaded.mp4' });
    });
    expect(uploadVideoMock).toHaveBeenCalledTimes(1);
    expect(uploadVideoMock.mock.calls[0]?.[1]).toBe('none');
  });

  const renderMediaMenuField = async (onNodeContextMenu?: (event: React.MouseEvent) => void) => {
    const openItemContextMenu = vi.fn();
    const adapter = { openItemContextMenu } as unknown as GalleryUiAdapter;

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <DndContext>
              <GalleryUiProvider adapter={adapter}>
                <div onContextMenu={onNodeContextMenu}>
                  <WorkflowFieldInput template={VIDEO_TEMPLATE} value={{ video_name: 'clip.mp4' }} onChange={vi.fn()} />
                </div>
              </GalleryUiProvider>
            </DndContext>
          </QueryClientProvider>
        </ChakraProvider>
      );
    });
    await vi.waitFor(() => {
      expect(host.textContent).toContain('640x480 · 5s');
    });

    // The thumbnail itself errors in the test server; the well's picker button holds it.
    return {
      openItemContextMenu,
      well: host.querySelector<HTMLElement>('button[aria-label="widgets.gallery.picker.chooseVideo"]')!,
    };
  };

  it("opens the gallery's item menu for the field's media at the pointer, and leaves its picker's to the browser", async () => {
    const nodeMenu = vi.fn((event: React.MouseEvent) => event.defaultPrevented);
    pickerState.isOpen = true;

    try {
      const { openItemContextMenu, well } = await renderMediaMenuField(nodeMenu);
      const rightClick = async (element: Element) => {
        const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 30, clientY: 40 });
        await act(() => element.dispatchEvent(event));
        return event;
      };

      expect((await rightClick(well)).defaultPrevented).toBe(true);
      // The node reads this to stay out of the way.
      expect(nodeMenu).toHaveLastReturnedWith(true);
      expect(openItemContextMenu).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({
          item: expect.objectContaining({ kind: 'video', name: 'clip.mp4' }),
          ref: { kind: 'video', name: 'clip.mp4' },
          x: 30,
          y: 40,
        })
      );

      expect(
        (await rightClick(document.querySelector('input[aria-label="Search the picker"]')!)).defaultPrevented
      ).toBe(false);
      expect(openItemContextMenu).toHaveBeenCalledOnce();
    } finally {
      pickerState.isOpen = false;
    }
  });

  it('anchors a keyboard-opened item menu below the start of the focused media, on screen, not at the pointer', async () => {
    const { openItemContextMenu, well } = await renderMediaMenuField();
    const elsewhere = document.createElement('div');
    const openFromKeyboard = async () => {
      await act(() => userEvent.keyboard('{Shift>}{F10}{/Shift}'));
      return openItemContextMenu.mock.lastCall?.[0] as { x: number; y: number };
    };

    elsewhere.style.cssText = 'position:fixed;right:0;bottom:0;width:20px;height:20px';
    host.append(elsewhere);
    await userEvent.hover(elsewhere);
    well.focus();
    const rect = well.getBoundingClientRect();

    expect(await openFromKeyboard()).toMatchObject({ x: rect.left, y: rect.bottom });

    // Media can run past the viewport where scrolling cannot reveal it, as a node on the transformed workflow canvas.
    host.style.cssText = `position:fixed;inset-inline:0;top:${window.innerHeight - rect.height / 2}px`;
    const moved = well.getBoundingClientRect();

    expect(moved.bottom).toBeGreaterThan(window.innerHeight);
    expect(await openFromKeyboard()).toMatchObject({ x: moved.left, y: window.innerHeight });
  });
});

describe('WorkflowFieldInput video frame picker', () => {
  // The mock backend's clip: 10 frames at 10 fps, one second long. Vite serves the package root.
  const FIXTURE_VIDEO = {
    ...SELECTED_GALLERY_VIDEO,
    durationSeconds: 1,
    fps: 10,
    fullUrl: '/scripts/mock-assets/fixture-video.mp4',
  };
  const i18n = createInstance();
  beforeAll(async () => {
    // Through the provider only: registering it globally would translate the raw keys later suites assert on.
    await i18n.init({
      interpolation: { escapeValue: false },
      lng: 'en',
      resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
    });
  });
  beforeEach(() => {
    resolveItemMock.mockResolvedValue(FIXTURE_VIDEO);
    graphNodes.push(makeFrameNode({ video_name: 'clip.mp4' }));
  });

  const renderFrameField = async (value: unknown, onChange: (value: unknown) => void) => {
    await act(() => {
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <QueryClientProvider client={queryClient}>
              <DndContext>
                <WorkflowFieldInput
                  nodeId="frame-node"
                  template={FRAME_INDEX_TEMPLATE}
                  value={value}
                  onChange={onChange}
                />
              </DndContext>
            </QueryClientProvider>
          </ChakraProvider>
        </I18nextProvider>
      );
    });
  };

  const getVideo = async (): Promise<HTMLVideoElement> => {
    await expect.element(page.getByRole('button', { name: 'Loop' })).toBeEnabled();

    return host.querySelector('video')!;
  };

  /** Waits until the player rests, paused, on `frame` (seeked to the middle of its display interval). */
  const expectShownFrame = async (video: HTMLVideoElement, frame: number) => {
    await expect
      .poll(() => video.paused && !video.seeking && Math.abs(video.currentTime - (frame + 0.5) / 10) < 0.01)
      .toBe(true);
  };

  const frameButton = () => page.getByRole('button', { name: 'Use this frame' });
  const expectUseFrameAvailable = async (available: boolean) => {
    if (available) {
      await expect.element(frameButton()).not.toHaveAttribute('aria-disabled');
    } else {
      await expect.element(frameButton()).toHaveAttribute('aria-disabled', 'true');
    }
  };

  const stepPlayerFrames = async (key: ',' | '.', count: number) => {
    const player = page.getByRole('group', { name: 'Frame preview' }).element() as HTMLElement;

    await act(() => player.focus());
    for (let step = 0; step < count; step += 1) {
      await userEvent.keyboard(key);
    }
  };

  it("shows the field's frame in a compact player with the video's poster and frame count", async () => {
    await renderFrameField(3, vi.fn());
    const video = await getVideo();

    expect(video.src).toContain('/scripts/mock-assets/fixture-video.mp4');
    expect(video.poster).toContain('/api/v1/videos/i/clip.mp4/thumbnail');
    await expectShownFrame(video, 3);
    await expect.element(page.getByText('3 / 9')).toBeVisible();
    await expectUseFrameAvailable(false);
  });

  it('commits the paused frame with "Use this frame", keeping focus once it is unavailable', async () => {
    const onChange = vi.fn();
    await renderFrameField(2, onChange);
    const video = await getVideo();
    await expectShownFrame(video, 2);
    await expectUseFrameAvailable(false);

    await stepPlayerFrames('.', 3);
    await expectShownFrame(video, 5);
    await expectUseFrameAvailable(true);
    const button = frameButton().element() as HTMLButtonElement;
    await act(() => button.focus());
    await userEvent.keyboard('{Enter}');

    expect(onChange).toHaveBeenCalledExactlyOnceWith(5);

    await renderFrameField(5, onChange);
    await expectShownFrame(video, 5);
    await expectUseFrameAvailable(false);
    await expect.element(frameButton()).toHaveAccessibleDescription('Pause on a different frame to use it');
    expect(document.activeElement).toBe(button);
    expect(button.closest('.nokey')).not.toBeNull();

    await userEvent.keyboard('{Enter}');
    await act(() => button.click());
    expect(onChange).toHaveBeenCalledOnce();
  });
});

describe('WorkflowFieldInput board field', () => {
  const BOARD_TEMPLATE = {
    name: 'board',
    title: 'Board',
    type: { cardinality: 'SINGLE', name: 'BoardField' },
  } as unknown as FieldInputTemplate;
  const boardDto = (boardId: string, overrides: Record<string, unknown> = {}) => ({
    admin_override: false,
    allowed_actions: ['read', 'add_media', 'manage_items', 'rename', 'set_cover', 'archive', 'move', 'delete'],
    archived: false,
    asset_count: 0,
    board_id: boardId,
    board_name: boardId,
    created_at: '2026-01-01T00:00:00.000Z',
    image_count: 1,
    is_inbox: false,
    my_role: 'owner',
    owner_username: 'Me',
    project_id: null,
    video_count: 0,
    ...overrides,
  });
  const listing = [
    boardDto('mine'),
    boardDto('team', { allowed_actions: ['read', 'add_media'], my_role: 'contributor', owner_username: 'Ana' }),
    boardDto('viewed', { allowed_actions: ['read'], my_role: 'viewer', owner_username: 'Ben' }),
  ];
  const i18n = createInstance();
  beforeAll(async () => {
    // Through the provider only: registering it globally would translate the raw keys later suites assert on.
    await i18n.init({
      interpolation: { escapeValue: false },
      lng: 'en',
      resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
    });
  });
  const renderBoardField = async (
    boardSharing: boolean | undefined,
    galleryHost: Partial<GalleryUiAdapter> = {},
    onChange: (value: unknown) => void = vi.fn()
  ) => {
    const adapter = { boardSharing, ...galleryHost } as unknown as GalleryUiAdapter;
    await act(() => {
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <QueryClientProvider client={queryClient}>
              <GalleryUiProvider adapter={adapter}>
                <WorkflowFieldInput template={BOARD_TEMPLATE} value="auto" onChange={onChange} />
              </GalleryUiProvider>
            </QueryClientProvider>
          </ChakraProvider>
        </I18nextProvider>
      );
    });
    const trigger = host.querySelector<HTMLElement>('[data-scope="select"][data-part="trigger"]');
    if (!trigger) {
      throw new Error('Board trigger not rendered');
    }
    await vi.waitFor(() => {
      expect(queryClient.getQueryCache().findAll({ queryKey: ['gallery', 'boards'] })[0]?.state.status).toBe('success');
    });
    await act(() => userEvent.click(trigger));
    const texts = (part: string) =>
      Array.from(document.querySelectorAll<HTMLElement>(`[data-scope="select"][data-part="${part}"]`))
        .map((element) => element.textContent?.trim())
        .filter(Boolean);

    return { groups: texts('item-group-label'), items: texts('item') };
  };

  beforeEach(() => {
    workflowApiMock.apiFetchJson.mockResolvedValue(listing);
  });

  afterEach(() => {
    delete galleryValues.autoAddBoardId;
    delete galleryValues.projectBoardId;
    delete galleryValues.selectedBoardId;
  });

  it("offers the boards the caller may add to under the gallery's own groups", async () => {
    expect(await renderBoardField(true)).toEqual({
      groups: ['Library', 'Shared with me'],
      items: ['Auto (Uncategorized)', 'Uncategorized', 'mine', 'team'],
    });
  });

  it('writes a picked board as its id', async () => {
    const onChange = vi.fn();

    await renderBoardField(true, {}, onChange);
    const mine = Array.from(document.querySelectorAll<HTMLElement>('[data-scope="select"][data-part="item"]')).find(
      (item) => item.textContent?.trim() === 'mine'
    )!;
    await act(() => userEvent.click(mine));

    expect(onChange).toHaveBeenCalledWith({ board_id: 'mine' });
  });

  it('sends an upload to the project inbox when the selected board is not in the live listing', async () => {
    uploadImageMock.mockResolvedValue({ imageName: 'uploaded.png' });
    galleryValues.selectedBoardId = 'restricted';
    galleryValues.projectBoardId = 'inbox';
    workflowApiMock.apiFetchJson.mockResolvedValue([boardDto('inbox', { is_inbox: true, project_id: 'project-1' })]);
    const onChange = vi.fn();
    const imageTemplate = {
      name: 'image',
      title: 'Image',
      type: { name: 'ImageField' },
    } as unknown as FieldInputTemplate;

    try {
      await renderField(imageTemplate, undefined, onChange);
      await act(() => findButton('Upload').click());
      const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;
      const transfer = new DataTransfer();
      transfer.items.add(new File(['data'], 'a.png', { type: 'image/png' }));
      fileInput.files = transfer.files;
      await act(() => fileInput.dispatchEvent(new Event('change', { bubbles: true })));

      await vi.waitFor(() => {
        expect(onChange).toHaveBeenCalledWith({ image_name: 'uploaded.png' });
      });
      expect(workflowApiMock.apiFetchJson).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/boards/'),
        expect.anything()
      );
      expect(uploadImageMock.mock.calls[0]?.[1]).toBe('inbox');
    } finally {
      delete galleryValues.selectedBoardId;
      delete galleryValues.projectBoardId;
    }
  });
});

const StatefulLoRAField = ({ initial, onCommit }: { initial: unknown; onCommit: (value: unknown) => void }) => {
  const [value, setValue] = useState(initial);
  const onChange = useCallback(
    (next: unknown) => {
      setValue(next);
      onCommit(next);
    },
    [onCommit]
  );

  return <WorkflowFieldInput template={LORA_COLLECTION_TEMPLATE} value={value} onChange={onChange} />;
};

describe('WorkflowFieldInput LoRA collection', () => {
  const i18n = createInstance();
  beforeAll(async () => {
    await i18n.init({
      interpolation: { escapeValue: false },
      lng: 'en',
      resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
    });
  });
  const renderStatefulLoras = async (initial: unknown, onCommit: (value: unknown) => void) => {
    await act(() => {
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <QueryClientProvider client={queryClient}>
              <DndContext>
                <StatefulLoRAField initial={initial} onCommit={onCommit} />
              </DndContext>
            </QueryClientProvider>
          </ChakraProvider>
        </I18nextProvider>
      );
    });
    await vi.waitFor(
      () => {
        expect(modelSelectState.props).not.toBeNull();
      },
      { timeout: 5_000 }
    );
  };

  const renderLoras = async (value: unknown, onChange: (value: unknown) => void) => {
    await renderField(LORA_COLLECTION_TEMPLATE, value, onChange, undefined, {}, i18n);
    // The picker is lazy; wait for Suspense to resolve it before asserting on the mounted widget.
    await vi.waitFor(
      () => {
        expect(modelSelectState.props).not.toBeNull();
      },
      { timeout: 5_000 }
    );
  };

  it('scopes the picker to the LoRAs the node can apply and hides the ones already added', async () => {
    await renderLoras([loraEntry()], vi.fn());

    const props = modelSelectState.props!;

    expect(props.modelTypes).toEqual(['lora']);
    expect(Array.from(props.excludeKeys ?? [])).toEqual(['sd-lora']);
    expect(props.filter?.(SD_LORA)).toBe(true);
    expect(props.filter?.(SDXL_LORA)).toBe(false);
  });

  it('edits one weight and removes one entry without disturbing the rest', async () => {
    const onChange = vi.fn();
    const other = loraEntry({
      lora: { base: 'sd-1', hash: 'hash-other', key: 'other-lora', name: 'Other', type: 'lora' },
      weight: 1,
    });

    await renderLoras([loraEntry(), other], onChange);

    const weightInputs = Array.from(host.querySelectorAll<HTMLInputElement>('input'));

    expect(weightInputs).toHaveLength(2);
    expect(host.textContent).toContain('Detail Tweaker');
    expect(host.textContent).toContain('Other');

    await typeWeight(weightInputs[0]!, '0.5');
    expect(onChange).toHaveBeenCalledWith([loraEntry({ weight: 0.5 }), other]);

    const removeOther = host.querySelector<HTMLButtonElement>('button[aria-label="Remove Other"]')!;

    await act(() => removeOther.click());
    expect(onChange).toHaveBeenLastCalledWith([loraEntry()]);
  });

  it('keeps an out-of-range or cleared weight on screen and invalid instead of clamping or restoring it', async () => {
    const onChange = vi.fn();

    await renderStatefulLoras([loraEntry()], onChange);

    const weight = host.querySelector<HTMLInputElement>('input')!;

    await typeWeight(weight, '999');
    expect(onChange).toHaveBeenLastCalledWith([loraEntry({ weight: 999 })]);
    expect(weight.getAttribute('aria-invalid')).toBe('true');
    await blurWeight(weight);
    expect(weight.value).toBe('999');
    expect(weight.getAttribute('aria-invalid')).toBe('true');

    await typeWeight(weight, '2');
    expect(onChange).toHaveBeenLastCalledWith([loraEntry({ weight: 2 })]);
    expect(weight.getAttribute('aria-invalid')).toBeNull();

    await typeWeight(weight, '{Backspace}');
    expect(onChange).toHaveBeenLastCalledWith([loraEntry({ weight: null })]);
    expect(weight.value).toBe('');
    expect(weight.getAttribute('aria-invalid')).toBe('true');
    await blurWeight(weight);
    expect(host.querySelector<HTMLInputElement>('input')).toBe(weight);
    expect(weight.value).toBe('');
    expect(weight.getAttribute('aria-invalid')).toBe('true');
  });
});

describe('WorkflowFieldInput model identifiers', () => {
  const ltx2Template = (uiModelFormat: string[] | null) =>
    ({
      name: 'model',
      title: 'Model',
      type: { batch: false, cardinality: 'SINGLE', name: 'ModelIdentifierField' },
      uiModelBase: ['ltx-2'],
      uiModelFormat,
      uiModelType: ['main'],
    }) as unknown as FieldInputTemplate;
  const transformer = { base: 'ltx-2', format: 'checkpoint', key: 'dev', type: 'main' };
  const fullFolder = { base: 'ltx-2', components_only: false, format: 'diffusers', key: 'full', type: 'main' };
  const componentsFolder = { base: 'ltx-2', components_only: true, format: 'diffusers', key: 'parts', type: 'main' };

  const pickerFilter = async (template: FieldInputTemplate) => {
    await renderField(template, undefined, vi.fn());
    await vi.waitFor(
      () => {
        expect(modelSelectState.props).not.toBeNull();
      },
      { timeout: 5_000 }
    );
    return modelSelectState.props!.filter!;
  };

  it('keeps components-only folders out of a format-agnostic model field', async () => {
    const filter = await pickerFilter(ltx2Template(null));

    expect(filter(transformer)).toBe(true);
    expect(filter(fullFolder)).toBe(true);
    expect(filter(componentsFolder)).toBe(false);
    expect(filter({ ...transformer, base: 'wan' })).toBe(false);
  });
});

describe('WorkflowFieldInput seed inputs', () => {
  const settle = async (action: () => void) => {
    await act(async () => {
      action();
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  };
  const seedInput = () => host.querySelector<HTMLInputElement>('input[aria-label="Seed"]');
  const diceButton = () => host.querySelector<HTMLButtonElement>('button[aria-label="common.newSeed"]');
  const modeTrigger = () => host.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]');
  const menuItem = (mode: string) =>
    document.querySelector<HTMLElement>(`[role="menuitemradio"][data-value="${mode}"]`);

  it('puts the mode menu beside a seed input and commits the chosen mode without touching the value', async () => {
    const onChange = vi.fn();
    const onSeedModeChange = vi.fn();

    // Node width: the largest seed has to fit beside the stepper, the dice and the trigger without scrolling.
    host.style.width = '18rem';
    await renderField(SEED_TEMPLATE, 4_294_967_295, onChange, undefined, { onSeedModeChange, seedMode: 'fixed' });

    expect(seedInput()?.disabled).toBe(false);
    expect(modeTrigger()?.getAttribute('aria-label')).toBe('common.seedMode.label: common.seedMode.fixed');
    expect(modeTrigger()?.getBoundingClientRect().width).toBeLessThanOrEqual(144);
    expect((seedInput() as HTMLInputElement).scrollWidth).toBeLessThanOrEqual(
      (seedInput() as HTMLInputElement).clientWidth
    );
    expect(modeTrigger()?.closest('.nokey')).not.toBeNull();

    await settle(() => modeTrigger()?.click());

    expect(menuItem('fixed')?.getAttribute('aria-checked')).toBe('true');
    expect(menuItem('fixed')?.closest('.nokey')).not.toBeNull();

    await settle(() => menuItem('increment')?.click());

    expect(onSeedModeChange).toHaveBeenCalledWith('increment');
    expect(onChange).not.toHaveBeenCalled();
  });

  it('rolls a new seed from the dice and writes it to the field as an integer in range', async () => {
    const onChange = vi.fn();

    await renderField(SEED_TEMPLATE, 42, onChange, undefined, { onSeedModeChange: vi.fn(), seedMode: 'fixed' });
    await settle(() => diceButton()?.click());

    expect(onChange).toHaveBeenCalledTimes(1);
    const rolled = onChange.mock.calls[0]?.[0] as number;

    expect(Number.isInteger(rolled)).toBe(true);
    expect(rolled).toBeGreaterThanOrEqual(0);
    expect(rolled).toBeLessThanOrEqual(4_294_967_295);
  });
});

const fullTemplate = (
  typeName: 'FloatField' | 'IntegerField' | 'StringField',
  overrides: Partial<FieldInputTemplate> = {}
): FieldInputTemplate => ({
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name: 'value',
  options: null,
  required: true,
  title: 'Value',
  type: { batch: false, cardinality: 'SINGLE', name: typeName },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});

/** Echoes each commit back as the next prop, deferred as the Canvas rebuild does, so caret bugs reproduce. */
const StatefulField = ({
  echo = 'sync',
  initial,
  onCommit,
  template,
}: {
  echo?: 'deferred' | 'sync';
  initial: unknown;
  onCommit: (value: unknown) => void;
  template: FieldInputTemplate;
}) => {
  const [value, setValue] = useState(initial);
  const onChange = useCallback(
    (next: unknown) => {
      if (echo === 'deferred') {
        startTransition(() => setValue(next));
      } else {
        setValue(next);
      }
      onCommit(next);
    },
    [echo, onCommit]
  );
  const invalid = getWorkflowFieldInvalidReason({ isConnected: false, template, value }) !== null;

  return (
    <Field.Root invalid={invalid}>
      <WorkflowFieldInput invalid={invalid} template={template} value={value} onChange={onChange} />
    </Field.Root>
  );
};

describe('WorkflowFieldInput text and number entry', () => {
  let renderCount = 0;
  const renderStateful = async (template: FieldInputTemplate, initial: unknown, echo?: 'deferred' | 'sync') => {
    const onCommit = vi.fn();

    renderCount += 1;
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <StatefulField key={renderCount} echo={echo} initial={initial} template={template} onCommit={onCommit} />
        </ChakraProvider>
      );
    });

    const input = host.querySelector<HTMLInputElement | HTMLTextAreaElement>('input, textarea');

    if (!input) {
      throw new Error('Field input not rendered');
    }

    // The click lands past the end of a short value; End makes the starting caret explicit either way.
    await act(async () => {
      await userEvent.click(input);
      await userEvent.keyboard('{End}');
    });

    return { input, onCommit };
  };
  const keys = (sequence: string) =>
    act(async () => {
      await userEvent.keyboard(sequence);
    });
  const blur = (input: HTMLElement) =>
    act(() => {
      input.blur();
    });
  const caret = (input: HTMLInputElement | HTMLTextAreaElement) => [input.selectionStart, input.selectionEnd];
  const isInvalid = (input: HTMLElement) => input.getAttribute('aria-invalid') === 'true';
  const FLOAT = fullTemplate('FloatField');

  const TEXT = fullTemplate('StringField');

  describe('text', () => {
    it('keeps the caret at the edit point while each keystroke echoes back through the value', async () => {
      const { input, onCommit } = await renderStateful(TEXT, 'hello world');

      await keys('{Home}X');
      expect(input.value).toBe('Xhello world');
      expect(caret(input)).toEqual([1, 1]);

      await keys('{End}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowLeft}Y');
      expect(input.value).toBe('Xhello Yworld');
      expect(caret(input)).toEqual([8, 8]);

      await keys('{End}Z');
      expect(input.value).toBe('Xhello YworldZ');
      expect(caret(input)).toEqual([14, 14]);
      expect(onCommit).toHaveBeenLastCalledWith('Xhello YworldZ');
      expect(document.activeElement).toBe(input);
    });

    it('keeps every keystroke and the caret when the committed value echoes back late', async () => {
      const { input, onCommit } = await renderStateful(TEXT, 'hello world', 'deferred');

      // Two keystrokes inside one act: the second lands before the first commit has echoed back.
      await keys('{Home}{ArrowRight}ab');
      expect(input.value).toBe('habello world');
      expect(caret(input)).toEqual([3, 3]);
      expect(onCommit).toHaveBeenLastCalledWith('habello world');

      const number = await renderStateful(FLOAT, 0.5, 'deferred');

      await keys('{Home}12');
      expect(number.input.value).toBe('120.5');
      expect(number.onCommit).toHaveBeenLastCalledWith(120.5);
      await keys('3');
      expect(number.input.value).toBe('1230.5');
    });

    it('keeps a composed entry intact while each update echoes back, in the input and the textarea', async () => {
      const TEXTAREA = fullTemplate('StringField', { uiComponent: 'textarea' });

      for (const [template, echo] of [
        [TEXT, 'sync'],
        [TEXT, 'deferred'],
        [TEXTAREA, 'deferred'],
      ] as const) {
        const { input, onCommit } = await renderStateful(template, 'hello world', echo);

        await keys('{Home}{ArrowRight}{ArrowRight}');
        // Kana-style entry: every update replaces the candidate text; the commit inserts the converted form.
        await act(async () => {
          await server.commands.imeCompose([
            { kind: 'compose', text: 'n' },
            { kind: 'compose', text: 'に' },
            { kind: 'compose', text: 'にh' },
            { kind: 'compose', text: 'にほ' },
            { kind: 'compose', text: 'にほん' },
          ]);
        });
        expect(input.value).toBe('heにほんllo world');
        expect(caret(input)).toEqual([5, 5]);

        await act(async () => {
          await server.commands.imeCompose([{ kind: 'commit', text: '日本' }]);
        });
        expect(input.value).toBe('he日本llo world');
        expect(caret(input)).toEqual([4, 4]);
        expect(onCommit).toHaveBeenLastCalledWith('he日本llo world');

        await keys('!');
        expect(input.value).toBe('he日本!llo world');
        expect(onCommit).toHaveBeenLastCalledWith('he日本!llo world');
      }
    });
  });

  describe('float', () => {
    it('replaces a selected digit or the whole value without jumping after the first character', async () => {
      const { input, onCommit } = await renderStateful(FLOAT, 0.5);

      await keys('{Shift>}{ArrowLeft}{/Shift}6');
      expect(input.value).toBe('0.6');
      expect(onCommit).toHaveBeenLastCalledWith(0.6);

      await keys('7');
      expect(input.value).toBe('0.67');

      await keys('{Control>}a{/Control}0.6');
      expect(input.value).toBe('0.6');
      expect(onCommit).toHaveBeenLastCalledWith(0.6);

      await keys('{Control>}a{/Control}25');
      expect(input.value).toBe('25');
      expect(onCommit).toHaveBeenLastCalledWith(25);
      expect(document.activeElement).toBe(input);

      await act(async () => {
        await userEvent.dblClick(input);
      });
      await keys('4');
      expect(input.value).toBe('4');
      expect(onCommit).toHaveBeenLastCalledWith(4);
    });

    it('keeps decimal and sign drafts while they are incomplete', async () => {
      const fromZero = await renderStateful(FLOAT, 0);
      const commitsBeforeDot = fromZero.onCommit.mock.calls.length;

      await keys('.');
      expect(fromZero.onCommit.mock.calls.length).toBe(commitsBeforeDot);
      await keys('6');
      expect(fromZero.input.value).toBe('0.6');
      expect(fromZero.onCommit).toHaveBeenLastCalledWith(0.6);

      const fromEmpty = await renderStateful(FLOAT, undefined);

      await keys('.6');
      expect(fromEmpty.input.value).toBe('.6');
      expect(fromEmpty.onCommit).toHaveBeenLastCalledWith(0.6);
      await blur(fromEmpty.input);
      expect(fromEmpty.input.value).toBe('0.6');

      const signed = await renderStateful(FLOAT, undefined);

      await keys('-');
      expect(signed.input.value).toBe('');
      expect((signed.input as HTMLInputElement).validity.badInput).toBe(true);
      await keys('4');
      expect(signed.input.value).toBe('-4');
      expect(signed.onCommit).toHaveBeenLastCalledWith(-4);
      expect(isInvalid(signed.input)).toBe(false);

      await keys('{Control>}a{/Control}0.5{Home}-');
      expect(signed.input.value).toBe('-0.5');
      expect(signed.onCommit).toHaveBeenLastCalledWith(-0.5);
      await keys('{End}{Backspace}{Backspace}{Backspace}{Backspace}1.');
      await keys('5');
      expect(signed.input.value).toBe('1.5');
      expect(signed.onCommit.mock.calls.slice(-2).map(([value]) => value)).toEqual([1, 1.5]);

      await keys('{Control>}a{/Control}1e-3');
      expect(signed.input.value).toBe('1e-3');
      expect(signed.onCommit).toHaveBeenLastCalledWith(0.001);
      await blur(signed.input);
      expect(signed.input.value).toBe('0.001');
    });

    it('normalizes a valid draft on blur and keeps an incomplete one visible and invalid', async () => {
      const { input, onCommit } = await renderStateful(FLOAT, 0.5);

      await keys('0');
      expect(input.value).toBe('0.50');
      expect(onCommit).toHaveBeenLastCalledWith(0.5);
      await blur(input);
      expect(input.value).toBe('0.5');

      await act(async () => {
        await userEvent.click(input);
      });
      await keys('{Control>}a{/Control}-');
      expect(input.value).toBe('');
      expect(onCommit).toHaveBeenLastCalledWith(undefined);
      await blur(input);
      expect((input as HTMLInputElement).validity.badInput).toBe(true);
      expect(isInvalid(input)).toBe(true);
      expect(onCommit).toHaveBeenLastCalledWith(undefined);
    });
  });

  describe('integer and bounds', () => {
    it('marks out-of-range and off-step entries invalid but editable, then valid once corrected', async () => {
      const { input, onCommit } = await renderStateful(
        fullTemplate('FloatField', { exclusiveMaximum: 1, minimum: 0, multipleOf: 0.25 }),
        0.5
      );

      await keys('{Control>}a{/Control}1');
      expect(input.value).toBe('1');
      expect(isInvalid(input)).toBe(true);
      await keys('{Backspace}0.75');
      expect(input.value).toBe('0.75');
      expect(isInvalid(input)).toBe(false);

      await keys('{Backspace}');
      expect(input.value).toBe('0.7');
      expect(onCommit).toHaveBeenLastCalledWith(0.7);
      expect(isInvalid(input)).toBe(true);
      await keys('5');
      expect(input.value).toBe('0.75');
      expect(isInvalid(input)).toBe(false);

      await keys('{Home}-');
      expect(input.value).toBe('-0.75');
      expect(isInvalid(input)).toBe(true);
      await keys('{Home}{Delete}');
      expect(input.value).toBe('0.75');
      expect(isInvalid(input)).toBe(false);
    });
  });
});

describe('WorkflowFieldInput scalar collections', () => {
  let renderCount = 0;
  const list = (typeName: 'FloatField' | 'IntegerField' | 'StringField', overrides: Partial<FieldInputTemplate> = {}) =>
    fullTemplate(typeName, { type: { batch: false, cardinality: 'COLLECTION', name: typeName }, ...overrides });
  const renderList = async (template: FieldInputTemplate, initial: unknown, echo?: 'deferred' | 'sync') => {
    const onCommit = vi.fn();

    renderCount += 1;
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <StatefulField key={renderCount} echo={echo} initial={initial} template={template} onCommit={onCommit} />
        </ChakraProvider>
      );
    });

    return { onCommit };
  };
  // Browser tests render raw i18n keys, so interpolated row names collapse to one key: rows are found by position.
  const row = (index: number) =>
    host.querySelectorAll<HTMLInputElement>('input[aria-label="nodes.collectionItemLabel"]')[index - 1] ?? null;
  const rows = () => host.querySelectorAll('input[aria-label="nodes.collectionItemLabel"]').length;
  const button = (label: 'nodes.addItem' | 'common.clear') => findButton(label);
  const removeButton = (index: number) =>
    host.querySelectorAll<HTMLButtonElement>('button[aria-label="nodes.removeItem"]')[index - 1]!;
  const click = (element: HTMLElement) =>
    act(async () => {
      await userEvent.click(element);
    });
  const keys = (sequence: string) =>
    act(async () => {
      await userEvent.keyboard(sequence);
    });

  it('appends, edits, and removes rows while each keystroke echoes back late', async () => {
    const { onCommit } = await renderList(list('IntegerField'), [1, 2], 'deferred');

    expect(rows()).toBe(2);
    await click(button('nodes.addItem'));
    expect(onCommit).toHaveBeenLastCalledWith([1, 2, 0]);
    expect(rows()).toBe(3);

    const third = row(3)!;

    await click(third);
    await keys('{Control>}a{/Control}42');
    expect(third.value).toBe('42');
    expect(onCommit).toHaveBeenLastCalledWith([1, 2, 42]);
    await keys('{Home}7');
    expect(third.value).toBe('742');
    expect(third.selectionStart).toBeNull(); // number inputs expose no caret; focus is the tell
    expect(document.activeElement).toBe(third);
    expect(onCommit).toHaveBeenLastCalledWith([1, 2, 742]);

    await click(removeButton(2));
    expect(onCommit).toHaveBeenLastCalledWith([1, 742]);
    expect(rows()).toBe(2);
    expect(row(2)?.value).toBe('742');
  });

  it('keeps keyboard focus in the list when a removal or clear unmounts the focused button', async () => {
    const { onCommit } = await renderList(list('IntegerField'), [1, 2, 3]);

    await act(() => removeButton(3).focus());
    await keys('{Enter}');
    expect(onCommit).toHaveBeenLastCalledWith([1, 2]);
    expect(document.activeElement).toBe(removeButton(2));

    await act(() => removeButton(1).focus());
    await keys('{Enter}');
    expect(onCommit).toHaveBeenLastCalledWith([2]);
    expect(document.activeElement).toBe(removeButton(1));

    await keys('{Enter}');
    expect(onCommit).toHaveBeenLastCalledWith([]);
    expect(document.activeElement).toBe(button('nodes.addItem'));

    await keys('{Enter}{Enter}');
    expect(rows()).toBe(2);
    await act(() => button('common.clear').focus());
    await keys('{Enter}');
    expect(rows()).toBe(0);
    expect(document.activeElement).toBe(button('nodes.addItem'));
  });
});

describe('WorkflowFieldInput record pickers', () => {
  // The pickers are a lazy chunk; load it up front so the first test is not timing a cold transform.
  beforeAll(async () => {
    await import('./RecordPickerInput');
  });

  const STYLE_PRESET = fullTemplate('StringField', {
    title: 'Style Preset',
    type: { batch: false, cardinality: 'SINGLE', name: 'StylePresetField' },
  });

  const presetDto = (id: string, name: string) => ({
    id,
    image: null,
    is_public: false,
    name,
    preset_data: { negative_prompt: '', positive_prompt: '{prompt}, cinematic' },
    type: 'user',
    user_id: 'user-1',
  });

  const combobox = () => host.querySelector<HTMLInputElement>('input[role="combobox"]');
  const option = (label: string) =>
    Array.from(document.querySelectorAll<HTMLElement>('[data-scope="combobox"][data-part="item"]')).find((item) =>
      item.textContent?.includes(label)
    );

  it('picks a style preset by name and clears it back to an absent value', async () => {
    workflowApiMock.apiFetchJson.mockResolvedValue([
      presetDto('preset-1', 'Cinematic'),
      presetDto('preset-2', 'Anime'),
    ]);
    const onChange = vi.fn();

    await renderField(STYLE_PRESET, undefined, onChange);
    await vi.waitFor(() => expect(combobox()?.placeholder).toBe('nodes.stylePresetSearch'));
    expect(host.querySelector('button[aria-label="nodes.stylePresetClear"]')).toBeNull();

    await act(() => userEvent.click(combobox()!));
    await vi.waitFor(() => expect(option('Anime')).toBeDefined());
    await act(() => userEvent.click(option('Anime')!));
    expect(onChange).toHaveBeenLastCalledWith({ style_preset_id: 'preset-2' });

    await renderField(STYLE_PRESET, { style_preset_id: 'preset-2' }, onChange);
    await vi.waitFor(() => expect(combobox()?.value).toBe('Anime'));
    expect(combobox()?.getAttribute('aria-invalid')).toBeNull();
    await act(() =>
      userEvent.click(host.querySelector<HTMLButtonElement>('button[aria-label="nodes.stylePresetClear"]')!)
    );
    expect(onChange).toHaveBeenLastCalledWith(undefined);
    expect(document.activeElement).toBe(combobox());
  });
});

describe('WorkflowFieldInput generators', () => {
  // The generator editor is a lazy chunk; load it up front so the first test is not timing a cold transform.
  beforeAll(async () => {
    await import('./GeneratorFieldInput');
  });

  const FLOAT_GENERATOR = fullTemplate('FloatField', {
    default: { count: 10, start: 0, step: 0.1, type: 'float_generator_arithmetic_sequence' },
    input: 'direct',
    title: 'Generator Type',
    type: { batch: false, cardinality: 'SINGLE', name: 'FloatGeneratorField' },
  });
  const STRING_GENERATOR = fullTemplate('StringField', {
    default: { input: 'foo,bar,baz,qux', splitOn: ',', type: 'string_generator_parse_string' },
    input: 'direct',
    title: 'Generator Type',
    type: { batch: false, cardinality: 'SINGLE', name: 'StringGeneratorField' },
  });
  const IMAGE_GENERATOR = fullTemplate('StringField', {
    default: { category: 'images', type: 'image_generator_images_from_board' },
    input: 'direct',
    title: 'Generator Type',
    type: { batch: false, cardinality: 'SINGLE', name: 'ImageGeneratorField' },
  });
  let renderCount = 0;
  const renderGenerator = async (template: FieldInputTemplate, initial: unknown) => {
    const onCommit = vi.fn();

    renderCount += 1;
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <StatefulField key={renderCount} initial={initial} template={template} onCommit={onCommit} />
          </QueryClientProvider>
        </ChakraProvider>
      );
    });
    await vi.waitFor(() => expect(host.querySelector('[data-scope="select"][data-part="trigger"]')).not.toBeNull());

    return { onCommit };
  };
  const numberInput = (suffix: string) =>
    host.querySelector<HTMLInputElement>(`input[type="number"][id$="${suffix}-number-input"]`)!;
  const selectVariant = async (label: string) => {
    await act(() => userEvent.click(host.querySelector('[data-scope="select"][data-part="trigger"]')!));
    const item = Array.from(document.querySelectorAll<HTMLElement>('[data-scope="select"][data-part="item"]')).find(
      (candidate) => candidate.textContent?.includes(label)
    )!;
    await act(() => userEvent.click(item));
  };

  it('previews an arithmetic sequence, commits edited settings as typed, and resets on a variant switch', async () => {
    const { onCommit } = await renderGenerator(FLOAT_GENERATOR, {
      count: 3,
      start: 1,
      step: 0.5,
      type: 'float_generator_arithmetic_sequence',
    });

    expect(host.textContent).toContain('1, 1.5, 2');

    await act(() => numberInput('-count').focus());
    await act(() => userEvent.keyboard('{Control>}a{/Control}4'));
    expect(onCommit).toHaveBeenLastCalledWith({
      count: 4,
      start: 1,
      step: 0.5,
      type: 'float_generator_arithmetic_sequence',
    });
    await vi.waitFor(() => expect(host.textContent).toContain('1, 1.5, 2, 2.5'), { timeout: 2000 });

    await selectVariant('nodes.parseString');
    expect(onCommit).toHaveBeenLastCalledWith({
      input: '0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1',
      splitOn: ',',
      type: 'float_generator_parse_string',
    });
  });

  it('shows a random draw as a count until a seed pins it, and refuses a count the queue cannot take', async () => {
    const { onCommit } = await renderGenerator(FLOAT_GENERATOR, {
      count: 4,
      max: 9,
      min: 2,
      seed: null,
      type: 'float_generator_random_distribution_uniform',
    });

    expect(host.textContent).toContain('<nodes.generatorNRandomValues>');
    expect(numberInput('-seed').disabled).toBe(true);
    expect(numberInput('-seed').getAttribute('aria-label')).toBe('nodes.generatorSeed');

    await act(() => numberInput('-count').focus());
    await act(() => userEvent.keyboard('{Control>}a{/Control}0'));
    expect(numberInput('-count').value).toBe('0');
    expect(onCommit).toHaveBeenLastCalledWith({
      count: 0,
      max: 9,
      min: 2,
      seed: null,
      type: 'float_generator_random_distribution_uniform',
    });
    expect(numberInput('-count').getAttribute('aria-invalid')).toBe('true');
    expect(numberInput('-min').getAttribute('aria-invalid')).toBeNull();
    expect(host.textContent).toContain('Count must be at least 1.');
    expect(numberInput('-min').value).toBe('2');
    await act(() => userEvent.keyboard('5'));
    expect(onCommit).toHaveBeenLastCalledWith({
      count: 5,
      max: 9,
      min: 2,
      seed: null,
      type: 'float_generator_random_distribution_uniform',
    });

    await act(() => userEvent.click(host.querySelector('[data-scope="checkbox"][data-part="control"]')!));
    expect(onCommit).toHaveBeenLastCalledWith({
      count: 5,
      max: 9,
      min: 2,
      seed: 0,
      type: 'float_generator_random_distribution_uniform',
    });
    expect(numberInput('-seed').disabled).toBe(false);
    expect(host.textContent).not.toContain('<nodes.generatorNRandomValues>');
    expect(host.textContent).toMatch(/\d+(\.\d+)?, \d+(\.\d+)?/);

    await act(() => numberInput('-count').focus());
    await act(() => userEvent.keyboard('{Control>}a{/Control}20000'));
    await vi.waitFor(() => expect(host.textContent).toContain('nodes.generatorTooMany'), { timeout: 2000 });

    await act(() => userEvent.keyboard('{Control>}a{/Control}{Backspace}'));
    expect(onCommit).toHaveBeenLastCalledWith({
      count: null,
      max: 9,
      min: 2,
      seed: 0,
      type: 'float_generator_random_distribution_uniform',
    });
    expect(host.textContent).toContain('Count is empty.');
    await act(() => numberInput('-count').blur());
    expect(numberInput('-count').value).toBe('');
    expect(numberInput('-count').getAttribute('aria-invalid')).toBe('true');

    await act(() => numberInput('-seed').focus());
    await act(() => userEvent.keyboard('{Control>}a{/Control}2.5'));
    expect(onCommit).toHaveBeenLastCalledWith(expect.objectContaining({ seed: 2.5 }));
    expect(numberInput('-seed').getAttribute('aria-invalid')).toBe('true');
    expect(numberInput('-seed').disabled).toBe(false);
  });

  it('previews dynamic prompts from the backend and marks a board the gallery no longer has', async () => {
    workflowApiMock.apiFetchJson.mockImplementation((path: string) =>
      path.includes('dynamicprompts')
        ? Promise.resolve({ error: null, prompts: ['a cute dog', 'a cute cat'] })
        : Promise.resolve([
            {
              board_id: 'b1',
              board_name: 'Portraits',
              image_count: 2,
              asset_count: 0,
              video_count: 0,
              asset_video_count: 0,
              archived: false,
            },
          ])
    );
    await renderGenerator(STRING_GENERATOR, {
      input: 'a cute {dog|cat}',
      maxPrompts: 5,
      type: 'string_generator_dynamic_prompts_combinatorial',
    });
    await vi.waitFor(() => expect(host.textContent).toContain('a cute dog, a cute cat'));

    await renderGenerator(IMAGE_GENERATOR, {
      board_id: 'gone',
      category: 'assets',
      type: 'image_generator_images_from_board',
    });
    await vi.waitFor(() =>
      expect(host.querySelector<HTMLInputElement>('input[role="combobox"]')?.value).toBe('nodes.generatorBoardMissing')
    );
    expect(host.querySelector('input[role="combobox"]')?.getAttribute('aria-invalid')).toBe('true');
  });
});
