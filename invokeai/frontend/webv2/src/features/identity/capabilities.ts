import { useAuthSession, type AuthSession } from './session';

export interface Capabilities {
  /** Server-wide runtime config (e.g. which GPUs generate); `PATCH /app/runtime_config` is admin-only. */
  canManageAppConfig: boolean;
  /** The image map's supplementary cluster-label vocabulary; the PUT route is admin-only. */
  canManageImageMapVocabulary: boolean;
  canManageModels: boolean;
  canManageNodes: boolean;
  /** Bulk import/export of prompt templates; the routes are admin-only. */
  canManagePromptTemplates: boolean;
  canManageUsers: boolean;
}

export const getCapabilities = (session: AuthSession): Capabilities => {
  if (session.phase !== 'ready') {
    return {
      canManageAppConfig: false,
      canManageImageMapVocabulary: false,
      canManageModels: false,
      canManageNodes: false,
      canManagePromptTemplates: false,
      canManageUsers: false,
    };
  }

  const isSingleUser = !session.multiuserEnabled;
  const isAdmin = isSingleUser || session.user?.is_admin === true;

  return {
    canManageAppConfig: isAdmin,
    canManageImageMapVocabulary: isAdmin,
    canManageModels: isAdmin,
    canManageNodes: isAdmin,
    // Matches the routers' `AdminUserOrDefault`: everyone qualifies in
    // single-user mode, only admins once multiuser is on.
    canManagePromptTemplates: isAdmin,
    canManageUsers: session.multiuserEnabled && session.user?.is_admin === true,
  };
};

export const useCapabilities = (): Capabilities => getCapabilities(useAuthSession());
