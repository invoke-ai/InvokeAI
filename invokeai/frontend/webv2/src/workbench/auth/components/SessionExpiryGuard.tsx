import { useNavigate } from '@tanstack/react-router';
import { useEffect } from 'react';

import { useAuthSession } from '@workbench/auth/session';

/**
 * Watches for mid-session token rejection (a 401 on any authenticated request)
 * and routes back to the login screen, which explains the expiry. Mounted once
 * inside the workbench route; renders nothing.
 */
export const SessionExpiryGuard = () => {
  const session = useAuthSession();
  const navigate = useNavigate();

  useEffect(() => {
    if (session.multiuserEnabled && session.sessionExpired && session.user === null) {
      void navigate({ to: '/login' });
    }
  }, [navigate, session.multiuserEnabled, session.sessionExpired, session.user]);

  return null;
};
