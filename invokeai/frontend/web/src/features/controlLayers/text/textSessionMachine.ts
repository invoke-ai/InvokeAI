export type TextSessionStatus = 'idle' | 'pending' | 'editing' | 'committed';
type TextSessionEvent = 'BEGIN' | 'EDIT' | 'COMMIT' | 'CANCEL';

export const transitionTextSessionStatus = (status: TextSessionStatus, event: TextSessionEvent): TextSessionStatus => {
  switch (status) {
    case 'idle':
      if (event === 'BEGIN') {
        return 'pending';
      }
      return status;
    case 'pending':
      if (event === 'EDIT') {
        return 'editing';
      }
      if (event === 'CANCEL') {
        return 'idle';
      }
      return status;
    case 'editing':
      if (event === 'COMMIT') {
        return 'committed';
      }
      if (event === 'CANCEL') {
        return 'idle';
      }
      return status;
    case 'committed':
      if (event === 'BEGIN') {
        return 'pending';
      }
      if (event === 'CANCEL') {
        return 'idle';
      }
      return status;
    default:
      return status;
  }
};
