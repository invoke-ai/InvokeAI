import { apiFetchJson } from '@workbench/backend/http';

const INTERMEDIATES_URL = '/api/v1/images/intermediates';

export interface ClearIntermediatesStateInput {
  canClearIntermediates: boolean;
  hasActiveQueueWork: boolean;
  intermediatesCount: number | null;
}

export interface ClearIntermediatesState {
  disabled: boolean;
  reason?: string;
}

export interface ClearIntermediatesConfirmation {
  body: string;
  confirmLabel: string;
  title: string;
}

export const getIntermediatesCount = (): Promise<number> => apiFetchJson<number>(INTERMEDIATES_URL);

export const clearIntermediates = (): Promise<number> => apiFetchJson<number>(INTERMEDIATES_URL, { method: 'DELETE' });

export const getClearIntermediatesState = ({
  canClearIntermediates,
  hasActiveQueueWork,
  intermediatesCount,
}: ClearIntermediatesStateInput): ClearIntermediatesState => {
  if (!canClearIntermediates) {
    return { disabled: true, reason: 'Only administrators can clear intermediates.' };
  }

  if (hasActiveQueueWork) {
    return { disabled: true, reason: 'Wait for pending or running queue work to finish.' };
  }

  if (intermediatesCount === null) {
    return { disabled: true, reason: 'Loading intermediate count.' };
  }

  if (intermediatesCount === 0) {
    return { disabled: true, reason: 'There are no intermediates to clear.' };
  }

  return { disabled: false };
};

export const getClearIntermediatesConfirmation = (count: number): ClearIntermediatesConfirmation => ({
  body: `This permanently deletes ${count} temporary intermediate image${count === 1 ? '' : 's'} and clears canvas layers or staged images that may reference them. Final gallery images are not removed.`,
  confirmLabel: 'Clear intermediates',
  title: 'Clear intermediate images?',
});
