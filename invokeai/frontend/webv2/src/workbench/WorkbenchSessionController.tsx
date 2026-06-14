import { useNavigate } from '@tanstack/react-router';
import { useEffect, useRef } from 'react';

import { hydrateProjectFromServer } from './projects/syncedPersistence';
import { useWorkbenchDispatch, useWorkbenchHasHydrated, useWorkbenchSelector } from './WorkbenchContext';
import type { WorkbenchSearch } from './projects/session';

/**
 * Keeps a mounted editor in line with the /app search params. Renders nothing.
 *
 * Boot is not its job — `WorkbenchProvider` consumes the params present at
 * mount as load options. This covers what happens afterwards:
 *
 * - `?new` is stripped once hydration has consumed it, so a reload (or a
 *   bookmark) does not mint another draft project.
 * - `?project` changing while the editor is open (a deep link clicked from
 *   elsewhere) focuses the tab if it is open, or hydrates it from the
 *   library and opens it if not.
 */
export const WorkbenchSessionController = ({ search }: { search: WorkbenchSearch }) => {
  const state = useWorkbenchSelector((snapshot) => snapshot.state);
  const dispatch = useWorkbenchDispatch();
  const hasHydrated = useWorkbenchHasHydrated();
  const navigate = useNavigate();
  const latestStateRef = useRef(state);
  // The mount-time param was already handled by the boot load options.
  const handledProjectIdRef = useRef<string | null>(search.project ?? null);

  latestStateRef.current = state;

  const isNewRequested = search.new === true;

  useEffect(() => {
    if (!hasHydrated || !isNewRequested) {
      return;
    }

    void navigate({ replace: true, search: {}, to: '/app' });
  }, [hasHydrated, isNewRequested, navigate]);

  const requestedProjectId = search.project;

  useEffect(() => {
    if (!hasHydrated || !requestedProjectId || handledProjectIdRef.current === requestedProjectId) {
      return undefined;
    }

    handledProjectIdRef.current = requestedProjectId;

    if (latestStateRef.current.projects.some((project) => project.id === requestedProjectId)) {
      dispatch({ projectId: requestedProjectId, type: 'switchProject' });

      return undefined;
    }

    let isCancelled = false;

    void hydrateProjectFromServer(requestedProjectId).then((project) => {
      if (isCancelled) {
        return;
      }

      if (project) {
        dispatch({ project, type: 'openProject' });
      } else {
        dispatch({
          kind: 'info',
          message: 'The linked project does not exist on this account — it may have been deleted.',
          title: 'Project not found',
          type: 'recordNotice',
        });
      }
    });

    return () => {
      isCancelled = true;
    };
  }, [dispatch, hasHydrated, requestedProjectId]);

  return null;
};
