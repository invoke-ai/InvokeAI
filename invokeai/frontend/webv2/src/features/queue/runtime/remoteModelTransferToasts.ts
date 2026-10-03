import { toaster } from '@platform/ui';

const PREFIX = '[[IRW_MODEL_TRANSFER]]';

type Phase =
  | 'preparing'
  | 'waiting'
  | 'downloading'
  | 'installing'
  | 'verifying'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface RemoteModelTransferProgress {
  backendItemId: number;
  remoteIndex: number;
  modelHash: string;
  name: string;
  directory: boolean;
  phase: Phase;
  bytes: number;
  totalBytes: number;
  error?: string;
}

const MAX_TERMINAL_IDS = 256;

const PHASES = new Set<Phase>([
  'preparing',
  'waiting',
  'downloading',
  'installing',
  'verifying',
  'completed',
  'failed',
  'cancelled',
]);

/** Model-transfer events are emitted by the primary for the queue item's owner. */
export const parseRemoteModelTransferProgress = (message: string): RemoteModelTransferProgress | null => {
  if (!message.startsWith(PREFIX)) {
    return null;
  }
  let data: unknown;
  try {
    data = JSON.parse(message.slice(PREFIX.length));
  } catch {
    return null;
  }
  if (data === null || typeof data !== 'object' || Array.isArray(data)) {
    return null;
  }
  const value = data as Record<string, unknown>;
  if (
    !Number.isSafeInteger(value.backend_item_id) ||
    (value.backend_item_id as number) < 1 ||
    !Number.isSafeInteger(value.remote_index) ||
    (value.remote_index as number) < 1 ||
    (value.remote_index as number) > 64 ||
    typeof value.model_hash !== 'string' ||
    value.model_hash.length === 0 ||
    value.model_hash.length > 256 ||
    typeof value.name !== 'string' ||
    value.name.length === 0 ||
    value.name.length > 256 ||
    typeof value.directory !== 'boolean' ||
    typeof value.phase !== 'string' ||
    !PHASES.has(value.phase as Phase) ||
    !Number.isSafeInteger(value.bytes) ||
    (value.bytes as number) < 0 ||
    !Number.isSafeInteger(value.total_bytes) ||
    (value.total_bytes as number) < 0 ||
    (value.error !== undefined && (typeof value.error !== 'string' || value.error.length > 2048))
  ) {
    return null;
  }
  return {
    backendItemId: value.backend_item_id as number,
    remoteIndex: value.remote_index as number,
    modelHash: value.model_hash as string,
    name: value.name as string,
    directory: value.directory as boolean,
    phase: value.phase as Phase,
    bytes: value.bytes as number,
    totalBytes: value.total_bytes as number,
    ...(typeof value.error === 'string' ? { error: value.error } : {}),
  };
};

const readableBytes = (bytes: number): string => {
  if (bytes >= 1024 ** 3) {
    return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  }
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
};

const descriptionFor = (value: RemoteModelTransferProgress): string => {
  switch (value.phase) {
    case 'preparing':
      return value.directory
        ? 'Preparing model directory and file checksums…'
        : 'Preparing single-file model transfer…';
    case 'waiting':
      return 'Waiting for the remote model installer…';
    case 'downloading': {
      const progress =
        value.totalBytes > 0 ? `${Math.min(100, Math.floor((value.bytes / value.totalBytes) * 100))}% · ` : '';
      const size =
        value.totalBytes > 0
          ? `${readableBytes(value.bytes)} / ${readableBytes(value.totalBytes)}`
          : `${readableBytes(value.bytes)} transferred`;
      return `Transferring ${progress}${size}`;
    }
    case 'installing':
      return 'Installing and registering model on remote worker…';
    case 'verifying':
      return 'Verifying installed model hash…';
    case 'completed':
      return 'Model transferred, installed and verified.';
    case 'failed':
      return value.error ? `Model transfer failed: ${value.error.slice(0, 240)}` : 'Model transfer failed.';
    case 'cancelled':
      return 'Model transfer cancelled.';
  }
};

/** One updating toast per primary queue item, remote worker and model. */
export const createRemoteModelTransferToasts = () => {
  const active = new Set<string>();
  const terminal = new Set<string>();

  const receive = (value: RemoteModelTransferProgress): void => {
    const id = `irw-model-transfer:${value.backendItemId}:${value.remoteIndex}:${value.modelHash}`;
    if (terminal.has(id)) {
      return;
    }

    const isTerminal = value.phase === 'completed' || value.phase === 'failed' || value.phase === 'cancelled';
    const title = `Remote ${value.remoteIndex} · ${value.name}`;
    const description = descriptionFor(value);
    const type =
      value.phase === 'completed'
        ? 'success'
        : value.phase === 'failed'
          ? 'error'
          : value.phase === 'cancelled'
            ? 'info'
            : 'loading';
    const duration = isTerminal ? 6000 : Infinity;

    if (active.has(id)) {
      toaster.update(id, { title, description, type, duration });
    } else {
      toaster.create({ id, title, description, type, duration });
    }

    if (isTerminal) {
      active.delete(id);
      terminal.add(id);
      while (terminal.size > MAX_TERMINAL_IDS) {
        const oldest = terminal.values().next().value;
        if (oldest === undefined) {
          break;
        }
        terminal.delete(oldest);
      }
    } else {
      active.add(id);
    }
  };

  const dispose = (): void => {
    for (const id of active) {
      toaster.dismiss(id);
    }
    for (const id of terminal) {
      toaster.dismiss(id);
    }
    active.clear();
    terminal.clear();
  };

  return { receive, dispose };
};
