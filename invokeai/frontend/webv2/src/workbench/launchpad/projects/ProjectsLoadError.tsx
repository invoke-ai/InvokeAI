import { LoadError } from '@platform/ui/LoadError';
import { getProjectLibrary, refreshProjectLibrary, useProjectLibrarySelector } from '@workbench/projects/library';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * A failed project listing, with a retry: the whole state where there are no projects to show (`state`), or above the
 * projects an earlier read listed (`inline`), never beside the empty state. A retry that recovers removes this alert,
 * so `getRecoveryFocus` names where focus goes then (the first project, say) instead of the page.
 */
export const ProjectsLoadError = ({
  getRecoveryFocus,
  layout,
  mb,
}: {
  getRecoveryFocus: () => HTMLElement | null;
  layout: 'inline' | 'state';
  mb?: string;
}) => {
  const { t } = useTranslation();
  const error = useProjectLibrarySelector((snapshot) => snapshot.error);
  const [isRetrying, setIsRetrying] = useState(false);
  // The library keeps its error status until a refresh succeeds, so the button shows the attempt itself.
  const retry = useCallback(() => {
    setIsRetrying(true);
    return refreshProjectLibrary().finally(() => {
      setIsRetrying(false);
      if (getProjectLibrary().status === 'ready') {
        // After the commit that replaced this alert with the projects.
        requestAnimationFrame(() => getRecoveryFocus()?.focus({ preventScroll: true }));
      }
    });
  }, [getRecoveryFocus]);

  return (
    <LoadError
      detail={error}
      isRetrying={isRetrying}
      layout={layout}
      mb={mb}
      message={t('projects.failedToLoad')}
      size="md"
      onRetry={retry}
    />
  );
};

/** The first control inside `container`, else the container itself (which must then take `tabIndex={-1}`). */
export const firstControlIn = (container: HTMLElement | null): HTMLElement | null =>
  container?.querySelector<HTMLElement>('a[href], button:not([disabled])') ?? container;
