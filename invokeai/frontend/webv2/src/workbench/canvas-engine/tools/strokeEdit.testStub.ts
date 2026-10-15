import type { StrokeSession } from './strokeSession';
import type { StrokeCommittedEvent, StrokeEdit } from './tool';

/** An admission that always grows. */
export const ADMIT_ALL: Pick<StrokeEdit, 'grow'> = { grow: () => true };

/** Commits `session`, accepting its event, and returns the event (null when nothing was published). */
export const commitStroke = (session: StrokeSession): StrokeCommittedEvent | null => {
  let committed: StrokeCommittedEvent | null = null;
  session.commit((event) => {
    committed = event;
    return true;
  });
  return committed;
};

/** A stroke edit that records what it admits and publishes; `admitBytes` caps growth to simulate refusal. */
export const createRecordingStrokeEdit = (admitBytes = Infinity) => {
  const record = { admitted: 0, cancelled: 0, commits: [] as StrokeCommittedEvent[] };
  const edit: StrokeEdit = {
    cancel: () => {
      record.cancelled += 1;
    },
    commit: (event) => {
      record.commits.push(event);
      return true;
    },
    grow: (bytes) => {
      if (record.admitted + bytes > admitBytes) {
        return false;
      }
      record.admitted += bytes;
      return true;
    },
  };
  return { edit, record };
};
