import type { FieldInputTemplate } from '@features/workflow/contracts';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkflowFieldInput } from './WorkflowFieldInput';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const uploadImageMock = vi.fn();
const uploadVideoMock = vi.fn();

vi.mock('@features/gallery', () => ({
  galleryDestinations: { list: () => Promise.resolve([]) },
  galleryTransfers: {
    upload: (...args: unknown[]) => uploadImageMock(...args),
    uploadVideo: (...args: unknown[]) => uploadVideoMock(...args),
  },
}));

const galleryValues: Record<string, unknown> = {};
const projectSnapshot = { galleryValues, id: 'project-1' };

vi.mock('@features/workflow/ui/WorkflowUiContext', () => ({
  useWorkflowProjectSelector: (selector: (project: { galleryValues: Record<string, unknown> }) => unknown) =>
    selector(projectSnapshot),
  useWorkflowUi: () => ({ project: { getSnapshot: () => projectSnapshot } }),
}));

const TEXTAREA_TEMPLATE = {
  name: 'prompt',
  title: 'Prompt',
  type: { name: 'StringField' },
  uiComponent: 'textarea',
} as unknown as FieldInputTemplate;

const VIDEO_TEMPLATE = {
  name: 'video',
  title: 'Video',
  type: { name: 'VideoField' },
} as unknown as FieldInputTemplate;

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
  delete galleryValues.selectedImage;
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const queryClient = new QueryClient();

const renderField = async (template: FieldInputTemplate, value: unknown, onChange: (value: unknown) => void) => {
  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <DndContext>
            <WorkflowFieldInput template={template} value={value} onChange={onChange} />
          </DndContext>
        </QueryClientProvider>
      </ChakraProvider>
    );
  });
};

const findButton = (label: string): HTMLButtonElement => {
  const button = Array.from(host.querySelectorAll('button')).find((el) => el.textContent?.includes(label));

  if (!button) {
    throw new Error(`Button "${label}" not found`);
  }

  return button;
};

describe('WorkflowFieldInput textarea', () => {
  it('uses the accessible unbounded resizable textarea for prompt-like string fields', async () => {
    await renderField(TEXTAREA_TEMPLATE, 'hello', vi.fn());

    const textarea = host.querySelector<HTMLTextAreaElement>('textarea')!;
    const handle = host.querySelector<HTMLElement>('[role="separator"]')!;

    expect(getComputedStyle(textarea).height).toBe('96px');
    expect(getComputedStyle(textarea).fontFamily).toContain('monospace');
    expect(handle.getAttribute('aria-label')).toBe('Resize Prompt');
    expect(handle.getAttribute('aria-valuemin')).toBe('56');
    expect(handle.hasAttribute('aria-valuemax')).toBe(false);

    await act(() => handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));
    expect(getComputedStyle(textarea).height).toBe('108px');
  });
});

describe('WorkflowFieldInput media inputs', () => {
  it('renders a direct-input widget for VideoField instead of falling back to connection-only', async () => {
    await renderField(VIDEO_TEMPLATE, undefined, vi.fn());

    expect(host.textContent).not.toContain('Connection only');
    expect(host.textContent).toContain('No video set');
    expect(findButton('Use gallery selection').disabled).toBe(true);
    expect(findButton('Upload').disabled).toBe(false);
    expect(host.querySelector<HTMLInputElement>('input[type="file"]')?.accept).toBe('video/*');
  });

  it('adopts a selected gallery video and clears it', async () => {
    galleryValues.selectedImage = SELECTED_GALLERY_VIDEO;
    const onChange = vi.fn();

    await renderField(VIDEO_TEMPLATE, undefined, onChange);

    const useSelection = findButton('Use gallery selection');

    expect(useSelection.disabled).toBe(false);
    await act(() => useSelection.click());
    expect(onChange).toHaveBeenCalledWith({ video_name: 'clip.mp4' });

    await renderField(VIDEO_TEMPLATE, { video_name: 'clip.mp4' }, onChange);
    expect(host.textContent).toContain('clip.mp4');
    expect(host.querySelector('img')?.src).toContain('/api/v1/videos/i/clip.mp4/thumbnail');

    await act(() => findButton('Clear').click());
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it('does not offer a selected gallery image to a video field', async () => {
    galleryValues.selectedImage = { ...SELECTED_GALLERY_VIDEO, kind: 'image' };

    await renderField(VIDEO_TEMPLATE, undefined, vi.fn());

    expect(findButton('Use gallery selection').disabled).toBe(true);
  });

  it('keeps COLLECTION media fields connection-only (the widget would write a bare object into a list)', async () => {
    const collectionTemplate = {
      name: 'videos',
      title: 'Videos',
      type: { cardinality: 'COLLECTION', name: 'VideoField' },
    } as unknown as FieldInputTemplate;

    await renderField(collectionTemplate, undefined, vi.fn());

    expect(host.textContent).toContain('Connection only');
    expect(host.querySelector('input[type="file"]')).toBeNull();
  });

  it('rejects a file whose type does not match the field kind', async () => {
    const onChange = vi.fn();

    await renderField(VIDEO_TEMPLATE, undefined, onChange);

    const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    const transfer = new DataTransfer();

    transfer.items.add(new File(['data'], 'a.png', { type: 'image/png' }));
    fileInput.files = transfer.files;
    await act(() => fileInput.dispatchEvent(new Event('change', { bubbles: true })));

    expect(uploadVideoMock).not.toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();
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

  it('keeps the image field on the image upload path', async () => {
    uploadImageMock.mockResolvedValue({ imageName: 'uploaded.png' });
    const onChange = vi.fn();
    const imageTemplate = {
      name: 'image',
      title: 'Image',
      type: { name: 'ImageField' },
    } as unknown as FieldInputTemplate;

    await renderField(imageTemplate, undefined, onChange);

    expect(host.querySelector<HTMLInputElement>('input[type="file"]')?.accept).toBe('image/*');
    await act(() => findButton('Upload').click());

    const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    const transfer = new DataTransfer();

    transfer.items.add(new File(['data'], 'a.png', { type: 'image/png' }));
    fileInput.files = transfer.files;
    await act(() => fileInput.dispatchEvent(new Event('change', { bubbles: true })));

    await vi.waitFor(() => {
      expect(onChange).toHaveBeenCalledWith({ image_name: 'uploaded.png' });
    });
    expect(uploadVideoMock).not.toHaveBeenCalled();
  });
});
