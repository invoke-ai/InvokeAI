import { skipToken } from '@reduxjs/toolkit/query';
import { useGetImageNamesQuery } from 'services/api/endpoints/images';

/**
 * Returns { isChecking, containsStarred }
 *
 * This helper is used to check whether the board has starred images and thus should be protected from accidental deletion
 */
export const useBoardContainsStarred = (boardId?: string, enabled = false) => {
  const queryArgs =
    enabled && boardId ? { board_id: boardId, starred_first: true, order_dir: 'DESC' as const } : skipToken;
  // here we force "starred_first" option to true to populate "starred_count" value
  // this should have no impact on user's board view preferences

  const { currentData, isFetching } = useGetImageNamesQuery(queryArgs);

  return {
    isChecking: isFetching, // this will help us to wait some for the request to complete, not sure if it's necessary
    containsStarred: (currentData?.starred_count ?? 0) > 0,
  };
};
