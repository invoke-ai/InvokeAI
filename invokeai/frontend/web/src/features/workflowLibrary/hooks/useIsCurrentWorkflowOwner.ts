import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { selectWorkflowId } from 'features/nodes/store/selectors';
import { useMemo } from 'react';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import { useGetWorkflowQuery } from 'services/api/endpoints/workflows';

/**
 * Returns true if the current user can save the currently-loaded workflow directly (not as a copy).
 *
 * In single-user mode, this always returns true.
 * In multiuser mode, returns true when:
 * - The workflow has no ID (new, unsaved workflow — will open Save As)
 * - The current user is the owner of the workflow
 * - The current user is an admin
 */
export const useIsCurrentWorkflowOwner = (): boolean => {
  const workflowId = useAppSelector(selectWorkflowId);
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();
  const { data: workflowData } = useGetWorkflowQuery(workflowId ?? skipToken);

  return useMemo(() => {
    // In single-user mode there is no concept of ownership, so saving is always allowed.
    if (!setupStatus?.multiuser_enabled) {
      return true;
    }

    // No authenticated user — be permissive.
    if (!currentUser) {
      return true;
    }

    // No workflow ID means this is a new/unsaved workflow. Clicking "Save" will open the
    // Save As dialog, so we should not block it.
    if (!workflowId) {
      return true;
    }

    // API data not yet available — be permissive to avoid incorrect disabling during loading.
    if (!workflowData) {
      return true;
    }

    return workflowData.user_id === currentUser.user_id || currentUser.is_admin;
  }, [setupStatus?.multiuser_enabled, workflowId, workflowData, currentUser]);
};
