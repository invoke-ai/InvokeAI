import type { StrokeSession } from './strokeSession';
import type { StrokeCommittedEvent, StrokeEdit } from './tool';

export const ADMIT_ALL: Pick<StrokeEdit, 'grow'> = { grow: () => true };

export const commitStroke = (session: StrokeSession): StrokeCommittedEvent | null => {
  let committed: StrokeCommittedEvent | null = null;
  session.commit((event) => {
    committed = event;
    return true;
  });
  return committed;
};

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
