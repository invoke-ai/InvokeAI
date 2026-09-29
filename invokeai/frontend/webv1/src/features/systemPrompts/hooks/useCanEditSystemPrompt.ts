import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { useMemo } from 'react';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import type { SystemPromptRecordDTO } from 'services/api/endpoints/systemPrompts';

/**
 * Returns true when the current user may edit/delete the given system prompt.
 *
 * In single-user mode, always true.
 * In multiuser mode, true only when the user owns the prompt or is an admin.
 */
export const useCanEditSystemPrompt = (prompt: SystemPromptRecordDTO | undefined): boolean => {
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();

  return useMemo(() => {
    if (!prompt) {
      return false;
    }
    // Single-user installs have no ownership concept.
    if (!setupStatus?.multiuser_enabled) {
      return true;
    }
    // No authenticated user — be permissive (matches workflow library behaviour).
    if (!currentUser) {
      return true;
    }
    return prompt.user_id === currentUser.user_id || currentUser.is_admin;
  }, [prompt, setupStatus?.multiuser_enabled, currentUser]);
};
