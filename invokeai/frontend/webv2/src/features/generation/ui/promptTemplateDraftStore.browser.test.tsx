import { accountLifecycle } from '@platform/state/accountLifecycle';
import { act, useLayoutEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { PendingPromptTemplateDraft } from './promptTemplateDraftStore';

import { setPendingPromptTemplateDraft, useOnPendingPromptTemplateDraft } from './promptTemplateDraftStore';

const beforeMountDraft: PendingPromptTemplateDraft = {
  negativePrompt: 'before negative',
  positivePrompt: 'before positive',
};
const afterMountDraft: PendingPromptTemplateDraft = {
  negativePrompt: 'after negative',
  positivePrompt: 'after positive',
};

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Probe = ({ onDraft }: { onDraft: (draft: PendingPromptTemplateDraft) => void }) => {
  useOnPendingPromptTemplateDraft(onDraft);
  return null;
};

const LayoutEmitter = ({ draft }: { draft: PendingPromptTemplateDraft | null }) => {
  useLayoutEffect(() => {
    if (draft) {
      setPendingPromptTemplateDraft(draft);
    }
  }, [draft]);

  return null;
};

const LatestCallbackProbe = ({
  draft,
  label,
  onDelivery,
}: {
  draft: PendingPromptTemplateDraft | null;
  label: string;
  onDelivery: (value: string) => void;
}) => {
  useOnPendingPromptTemplateDraft((pending) => onDelivery(`${label}:${pending.positivePrompt}`));
  return <LayoutEmitter draft={draft} />;
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

describe('useOnPendingPromptTemplateDraft', () => {
  it('delivers pre-mount and live handoffs once', async () => {
    const onDraft = vi.fn();
    setPendingPromptTemplateDraft(beforeMountDraft);

    await act(() => root.render(<Probe onDraft={onDraft} />));
    await act(() => setPendingPromptTemplateDraft(afterMountDraft));
    await act(() => root.render(<Probe onDraft={onDraft} />));

    expect(onDraft.mock.calls).toEqual([[beforeMountDraft], [afterMountDraft]]);
  });

  it('uses the latest callback for a draft published during an update layout', async () => {
    const onDelivery = vi.fn();
    await act(() => root.render(<LatestCallbackProbe draft={null} label="old" onDelivery={onDelivery} />));

    await act(() => root.render(<LatestCallbackProbe draft={afterMountDraft} label="new" onDelivery={onDelivery} />));

    expect(onDelivery).toHaveBeenCalledOnce();
    expect(onDelivery).toHaveBeenCalledWith('new:after positive');
  });

  it('does not deliver account A draft after account B becomes active', async () => {
    const onDraft = vi.fn();
    accountLifecycle.activate('draft-a', ':user:draft-a');
    setPendingPromptTemplateDraft(beforeMountDraft);

    accountLifecycle.activate('draft-b', ':user:draft-b');
    await act(() => root.render(<Probe onDraft={onDraft} />));

    expect(onDraft).not.toHaveBeenCalled();
  });

  it('stops delivering after cleanup', async () => {
    const onDraft = vi.fn();
    await act(() => root.render(<Probe onDraft={onDraft} />));
    await act(() => root.render(null));

    setPendingPromptTemplateDraft(afterMountDraft);

    expect(onDraft).not.toHaveBeenCalled();

    // Drain the still-pending one-shot so the module store cannot leak into a
    // later test if the runner changes declaration order.
    await act(() => root.render(<Probe onDraft={vi.fn()} />));
  });
});
