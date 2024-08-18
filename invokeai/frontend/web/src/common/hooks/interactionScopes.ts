import { logger } from 'app/logging/logger';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { objectKeys } from 'common/util/objectKeys';
import { isEqual } from 'lodash-es';
import type { Atom } from 'nanostores';
import { atom, computed } from 'nanostores';
import type { RefObject } from 'react';
import { useEffect, useMemo } from 'react';

const log = logger('system');

const _INTERACTION_SCOPES = ['gallery', 'canvas', 'workflows', 'imageViewer'] as const;

type InteractionScope = (typeof _INTERACTION_SCOPES)[number];

export const $activeScopes = atom<Set<InteractionScope>>(new Set());

type InteractionScopeData = {
  targets: Set<HTMLElement>;
  $isActive: Atom<boolean>;
};

export const INTERACTION_SCOPES: Record<InteractionScope, InteractionScopeData> = _INTERACTION_SCOPES.reduce(
  (acc, region) => {
    acc[region] = {
      targets: new Set(),
      $isActive: computed($activeScopes, (activeScopes) => activeScopes.has(region)),
    };
    return acc;
  },
  {} as Record<InteractionScope, InteractionScopeData>
);

const formatScopes = (interactionScopes: Set<InteractionScope>) => {
  if (interactionScopes.size === 0) {
    return 'none';
  }
  return Array.from(interactionScopes).join(', ');
};

export const addScope = (scope: InteractionScope) => {
  const currentScopes = $activeScopes.get();
  if (currentScopes.has(scope)) {
    return;
  }
  const newScopes = new Set(currentScopes);
  newScopes.add(scope);
  $activeScopes.set(newScopes);
  log.trace(`Added scope ${scope}: ${formatScopes($activeScopes.get())}`);
};

export const removeScope = (scope: InteractionScope) => {
  const currentScopes = $activeScopes.get();
  if (!currentScopes.has(scope)) {
    return;
  }
  const newScopes = new Set(currentScopes);
  newScopes.delete(scope);
  $activeScopes.set(newScopes);
  log.trace(`Removed scope ${scope}: ${formatScopes($activeScopes.get())}`);
};

export const setScopes = (scopes: InteractionScope[]) => {
  const newScopes = new Set(scopes);
  $activeScopes.set(newScopes);
  log.trace(`Set scopes: ${formatScopes($activeScopes.get())}`);
};

export const clearScopes = () => {
  $activeScopes.set(new Set());
  log.trace(`Cleared scopes`);
};

export const useScopeOnFocus = (scope: InteractionScope, ref: RefObject<HTMLElement>) => {
  useEffect(() => {
    const element = ref.current;

    if (!element) {
      return;
    }

    INTERACTION_SCOPES[scope].targets.add(element);

    return () => {
      INTERACTION_SCOPES[scope].targets.delete(element);
    };
  }, [ref, scope]);
};

type UseScopeOnMountOptions = {
  mount?: boolean;
  unmount?: boolean;
};

const defaultUseScopeOnMountOptions: UseScopeOnMountOptions = {
  mount: true,
  unmount: true,
};

export const useScopeOnMount = (scope: InteractionScope, options?: UseScopeOnMountOptions) => {
  useEffect(() => {
    const { mount, unmount } = { ...defaultUseScopeOnMountOptions, ...options };

    if (mount) {
      addScope(scope);
    }

    return () => {
      if (unmount) {
        removeScope(scope);
      }
    };
  }, [options, scope]);
};

export const useScopeImperativeApi = (scope: InteractionScope) => {
  const api = useMemo(() => {
    return {
      add: () => {
        addScope(scope);
      },
      remove: () => {
        removeScope(scope);
      },
    };
  }, [scope]);

  return api;
};

const handleFocusEvent = (_event: FocusEvent) => {
  const activeElement = document.activeElement;
  if (!(activeElement instanceof HTMLElement)) {
    return;
  }

  const newActiveScopes = new Set<InteractionScope>();

  for (const scope of objectKeys(INTERACTION_SCOPES)) {
    for (const element of INTERACTION_SCOPES[scope].targets) {
      if (element.contains(activeElement)) {
        newActiveScopes.add(scope);
      }
    }
  }

  const oldActiveScopes = $activeScopes.get();
  if (!isEqual(oldActiveScopes, newActiveScopes)) {
    $activeScopes.set(newActiveScopes);
    log.trace(`Scopes changed: ${formatScopes($activeScopes.get())}`);
  }
};

export const useScopeFocusWatcher = () => {
  useAssertSingleton('useScopeFocusWatcher');

  useEffect(() => {
    window.addEventListener('focus', handleFocusEvent, true);
    return () => {
      window.removeEventListener('focus', handleFocusEvent, true);
    };
  }, []);
};
