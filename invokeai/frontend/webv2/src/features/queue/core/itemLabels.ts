import type { QueueItem } from './historyTypes';
import type { QueueItemReadModel } from './types';

import { extractGenerationMeta } from './generationMeta';
import { getQueueItemSnapshotPositivePrompt } from './historySnapshot';

/** Long enough to tell prompts apart in a toast or a row, short enough for one line. */
const LABEL_MAX_LENGTH = 60;

/** One line of at most `maxLength` characters, cut at a word where one is near. */
export const shortenQueueLabel = (text: string, maxLength = LABEL_MAX_LENGTH): string => {
  const line = text.replace(/\s+/g, ' ').trim();

  if (line.length <= maxLength) {
    return line;
  }

  const cut = line.slice(0, maxLength - 1);
  const wordEnd = cut.lastIndexOf(' ');

  return `${(wordEnd > maxLength * 0.6 ? cut.slice(0, wordEnd) : cut).trimEnd()}…`;
};

const readWorkflowName = (workflow: Record<string, unknown> | undefined | null): string => {
  const name = workflow?.name;

  return typeof name === 'string' ? name.trim() : '';
};

/**
 * What a backend queue item is, in the user's words: its workflow's name, else its prompt. Empty when the item carries
 * neither (another user's redacted item, a graph with no prompt), and callers fall back to its id.
 */
export const getQueueItemReadModelLabel = (item: QueueItemReadModel): string => {
  const workflowName = item.workflowName?.trim();

  if (workflowName) {
    return shortenQueueLabel(workflowName);
  }

  return shortenQueueLabel(extractGenerationMeta(item).positivePrompt ?? '');
};

/**
 * The same for a run this browser submitted, read from its submission snapshot. Read defensively, as the other
 * snapshot readers are: a run restored from an older journal may lack parts of it.
 */
export const getQueueItemSnapshotLabel = (item: QueueItem): string => {
  const submission = (item.snapshot as Partial<QueueItem['snapshot']> | undefined)?.backendSubmission;
  const workflowName = submission?.kind === 'workflow' ? readWorkflowName(submission.workflow) : '';

  return shortenQueueLabel(workflowName || getQueueItemSnapshotPositivePrompt(item));
};
