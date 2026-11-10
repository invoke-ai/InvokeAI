/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { FieldInputTemplate } from '@features/workflow/contracts';
import type { WorkflowFieldInputProps } from '@features/workflow/ui/fields/WorkflowFieldInput';

import { ChakraProvider } from '@chakra-ui/react';
import { MANAGED_FILE_ACCEPT } from '@features/workflow/data/managedFiles';
import { formatBytes } from '@platform/i18n/languages';
import { ApiError } from '@platform/transport/http';
import { toaster } from '@platform/ui/toaster';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkflowFieldInput } from './WorkflowFieldInput';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const httpMock = vi.hoisted(() => ({ apiFetch: vi.fn(), apiFetchJson: vi.fn() }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  apiFetch: httpMock.apiFetch,
  apiFetchJson: httpMock.apiFetchJson,
}));

type DocumentState = { edges: unknown[]; id: string; nodes: { id: string }[] };

const projectState = vi.hoisted(() => ({
  projectGraph: { edges: [], id: 'graph-1', nodes: [{ id: 'node-1' }] } as DocumentState,
  workflows: [] as { document: DocumentState }[],
}));

vi.mock('@features/workflow/ui/WorkflowUiContext', () => ({
  useWorkflowProjectSelector: (selector: (project: typeof projectState) => unknown) => selector(projectState),
  useWorkflowUi: () => ({ project: { getSnapshot: () => projectState } }),
}));

const FILE_TEMPLATE = {
  input: 'direct',
  name: 'file',
  title: 'File',
  type: { batch: false, cardinality: 'SINGLE', name: 'FileField' },
} as unknown as FieldInputTemplate;

const FILE_DTO = {
  content_type: 'text/markdown',
  created_at: '2026-06-25T04:53:34.000Z',
  file_id: 'file-1',
  file_name: 'notes.md',
  size_bytes: 2048,
};

const REPLACEMENT_DTO = {
  ...FILE_DTO,
  file_id: 'file-2',
  file_name: 'prompt.txt',
  size_bytes: 512,
};

let host: HTMLDivElement;
let root: Root;
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  httpMock.apiFetch.mockReset().mockResolvedValue(new Response(null, { status: 204 }));
  httpMock.apiFetchJson.mockReset();
  projectState.projectGraph = { edges: [], id: 'graph-1', nodes: [{ id: 'node-1' }] };
  projectState.workflows = [];
  queryClient.clear();
});

afterEach(async () => {
  vi.restoreAllMocks();
  await act(() => root.unmount());
  host.remove();
});

const renderField = async (
  value: unknown,
  onChange: (value: unknown) => void,
  props: Pick<WorkflowFieldInputProps, 'nodeId' | 'workflowId'> = { nodeId: 'node-1' }
) => {
  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <WorkflowFieldInput template={FILE_TEMPLATE} value={value} onChange={onChange} {...props} />
        </QueryClientProvider>
      </ChakraProvider>
    );
  });

  // The field widget is lazily imported, so wait for its fallback to be replaced.
  await vi.waitFor(() => expect(host.querySelector('input[type="file"]')).not.toBeNull());
};

const findButton = (label: string): HTMLButtonElement => {
  const button = Array.from(host.querySelectorAll('button')).find((el) => el.textContent?.includes(label));

  if (!button) {
    throw new Error(`Button "${label}" not found`);
  }

  return button;
};

const pickFile = async (file: File) => {
  const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;
  const transfer = new DataTransfer();

  transfer.items.add(file);
  fileInput.files = transfer.files;
  await act(() => fileInput.dispatchEvent(new Event('change', { bubbles: true })));
};

const dropFile = async (file: File) => {
  const dropZone = host.querySelector<HTMLElement>('[role="group"]')!;
  const transfer = new DataTransfer();

  transfer.items.add(file);
  await act(() =>
    dropZone.dispatchEvent(new DragEvent('dragenter', { bubbles: true, cancelable: true, dataTransfer: transfer }))
  );
  await act(() =>
    dropZone.dispatchEvent(new DragEvent('dragover', { bubbles: true, cancelable: true, dataTransfer: transfer }))
  );
  await act(() =>
    dropZone.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: transfer }))
  );
};

describe('WorkflowFieldInput managed file', () => {
  it('uploads a picked file and writes only its id into the field', async () => {
    httpMock.apiFetchJson.mockResolvedValue(FILE_DTO);
    const onChange = vi.fn();

    await renderField(undefined, onChange);
    const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;

    expect(fileInput.accept).toBe(MANAGED_FILE_ACCEPT);
    await pickFile(new File(['# Notes'], 'notes.md', { type: 'text/markdown' }));

    await vi.waitFor(() => expect(onChange).toHaveBeenCalledWith({ file_id: 'file-1' }));
    const [path, init] = httpMock.apiFetchJson.mock.calls[0]!;

    expect(path).toBe('/api/v1/files/upload');
    expect(init).toEqual(expect.objectContaining({ method: 'POST' }));
    expect((init.body as FormData).get('file')).toBeInstanceOf(File);
  });

  it('uploads a file dropped onto the zone', async () => {
    httpMock.apiFetchJson.mockResolvedValue(FILE_DTO);
    const onChange = vi.fn();

    await renderField(undefined, onChange);
    await dropFile(new File(['# Notes'], 'notes.md', { type: 'text/markdown' }));

    await vi.waitFor(() => expect(onChange).toHaveBeenCalledWith({ file_id: 'file-1' }));
    expect(httpMock.apiFetchJson.mock.calls[0]?.[0]).toBe('/api/v1/files/upload');
  });

  it('shows the stored file name and size, and clear removes only the reference', async () => {
    httpMock.apiFetchJson.mockResolvedValue(FILE_DTO);
    const onChange = vi.fn();

    await renderField({ file_id: 'file-1' }, onChange);

    await vi.waitFor(() => expect(host.textContent).toContain('notes.md'));
    expect(host.textContent).toContain(formatBytes(FILE_DTO.size_bytes));
    expect(httpMock.apiFetchJson).toHaveBeenCalledWith('/api/v1/files/i/file-1', {
      signal: expect.any(AbortSignal),
    });

    await act(() => findButton('common.clear').click());

    expect(onChange).toHaveBeenCalledWith(undefined);
    expect(httpMock.apiFetch).not.toHaveBeenCalled();
  });

  it('replaces a referenced file without deleting the previous one', async () => {
    httpMock.apiFetchJson.mockImplementation((path: string) =>
      Promise.resolve(path === '/api/v1/files/upload' ? REPLACEMENT_DTO : FILE_DTO)
    );
    const onChange = vi.fn();

    await renderField({ file_id: 'file-1' }, onChange);
    await vi.waitFor(() => expect(host.textContent).toContain('notes.md'));

    await pickFile(new File(['hello'], 'prompt.txt', { type: 'text/plain' }));

    await vi.waitFor(() => expect(onChange).toHaveBeenCalledWith({ file_id: 'file-2' }));
    expect(httpMock.apiFetch).not.toHaveBeenCalled();
  });

  it('offers clear and retry when the stored file cannot be read, without mutating the field', async () => {
    httpMock.apiFetchJson
      .mockRejectedValueOnce(new ApiError(JSON.stringify({ detail: 'File not found' }), 404))
      .mockResolvedValue(FILE_DTO);
    const onChange = vi.fn();

    await renderField({ file_id: 'file-1' }, onChange);

    await vi.waitFor(() => expect(host.textContent).toContain('nodes.fileUnavailable'));
    expect(onChange).not.toHaveBeenCalled();
    expect(findButton('common.clear')).not.toBeNull();
    expect(findButton('common.retry')).not.toBeNull();

    await act(() => findButton('common.retry').click());
    await vi.waitFor(() => expect(host.textContent).toContain('notes.md'));

    await act(() => findButton('common.clear').click());
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it('rejects a file type the server does not accept before uploading it', async () => {
    const createToast = vi.spyOn(toaster, 'create').mockImplementation(() => '');
    const onChange = vi.fn();

    await renderField(undefined, onChange);
    await pickFile(new File(['binary'], 'archive.zip', { type: 'application/zip' }));

    await vi.waitFor(() =>
      expect(createToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'nodes.fileUploadRejected', type: 'error' })
      )
    );
    expect(httpMock.apiFetchJson).not.toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();
    createToast.mockRestore();
  });

  it('shows the server’s own upload refusal message', async () => {
    const createToast = vi.spyOn(toaster, 'create').mockImplementation(() => '');
    httpMock.apiFetchJson.mockRejectedValue(
      new ApiError(JSON.stringify({ detail: 'File type is not supported' }), 415)
    );
    const onChange = vi.fn();

    await renderField(undefined, onChange);
    await pickFile(new File(['# Notes'], 'notes.md', { type: 'text/markdown' }));

    await vi.waitFor(() =>
      expect(createToast).toHaveBeenCalledWith(
        expect.objectContaining({ description: 'File type is not supported', title: 'nodes.fileUploadFailed' })
      )
    );
    expect(onChange).not.toHaveBeenCalled();
    createToast.mockRestore();
  });

  it('does not write a late upload into a document that was replaced, and removes the orphan', async () => {
    let resolveUpload: ((dto: typeof FILE_DTO) => void) | null = null;

    httpMock.apiFetchJson.mockImplementation(
      (path: string) =>
        new Promise((resolve) => {
          if (path === '/api/v1/files/upload') {
            resolveUpload = resolve;
            return;
          }

          resolve(FILE_DTO);
        })
    );
    const onChange = vi.fn();

    await renderField(undefined, onChange);
    await pickFile(new File(['# Notes'], 'notes.md', { type: 'text/markdown' }));
    await vi.waitFor(() => expect(resolveUpload).not.toBeNull());

    projectState.projectGraph = { edges: [], id: 'graph-2', nodes: [{ id: 'node-1' }] };
    await act(() => {
      resolveUpload?.(FILE_DTO);
    });

    await vi.waitFor(() =>
      expect(httpMock.apiFetch).toHaveBeenCalledWith(
        '/api/v1/files/i/file-1',
        expect.objectContaining({ method: 'DELETE' })
      )
    );
    expect(onChange).not.toHaveBeenCalled();
  });
});
