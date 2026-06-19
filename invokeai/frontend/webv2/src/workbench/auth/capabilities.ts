import { useAuthSession, type AuthSession } from './session';

export interface Capabilities {
  canManageModels: boolean;
  canManageNodes: boolean;
  canManageUsers: boolean;
}

export const getCapabilities = (session: AuthSession): Capabilities => {
  const isSingleUser = !session.multiuserEnabled;
  const isAdmin = isSingleUser || session.user?.is_admin === true;

  return {
    canManageModels: isAdmin,
    canManageNodes: isAdmin,
    canManageUsers: session.multiuserEnabled && session.user?.is_admin === true,
  };
};

export const useCapabilities = (): Capabilities => getCapabilities(useAuthSession());
