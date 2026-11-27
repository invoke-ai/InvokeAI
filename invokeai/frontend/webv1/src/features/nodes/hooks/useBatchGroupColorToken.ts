import { useMemo } from 'react';

export const useBatchGroupColorToken = (batchGroupId?: string) => {
  const batchGroupColorToken = useMemo(() => {
    switch (batchGroupId) {
      case 'Group 1':
        return 'invokeGreen.300';
      case 'Group 2':
        return 'invokeBlue.300';
      case 'Group 3':
        return 'invokePurple.200';
      case 'Group 4':
        return 'invokeRed.300';
      case 'Group 5':
        return 'invokeYellow.300';
      default:
        return undefined;
    }
  }, [batchGroupId]);

  return batchGroupColorToken;
};
