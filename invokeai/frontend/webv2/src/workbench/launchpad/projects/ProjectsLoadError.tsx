import { Alert } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { getProjectLibrary, refreshProjectLibrary, useProjectLibrarySelector } from '@workbench/projects/library';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * A failed project listing, with a retry: shown instead of the empty state, never beside it. A retry that recovers
 * removes this alert, so `getRecoveryFocus` names where focus goes then (the first project, say) instead of the page.
 */
export const ProjectsLoadError = ({
  getRecoveryFocus,
  mb,
}: {
  getRecoveryFocus: () => HTMLElement | null;
  mb?: string;
}) => {
  const { t } = useTranslation();
  const error = useProjectLibrarySelector((snapshot) => snapshot.error);
  const [isRetrying, setIsRetrying] = useState(false);
  // The library keeps its error status until a refresh succeeds, so the button shows the attempt itself.
  const retry = useCallback(() => {
    setIsRetrying(true);
    void refreshProjectLibrary().finally(() => {
      setIsRetrying(false);
      if (getProjectLibrary().status === 'ready') {
        // After the commit that replaced this alert with the projects.
        requestAnimationFrame(() => getRecoveryFocus()?.focus({ preventScroll: true }));
      }
    });
  }, [getRecoveryFocus]);

  return (
    <Alert.Root borderRadius="md" mb={mb} role="alert" status="error">
      <Alert.Indicator />
      <Alert.Content flex="1">
        <Alert.Title fontSize="md">{t('projects.failedToLoad')}</Alert.Title>
        {error ? <Alert.Description fontSize="md">{error}</Alert.Description> : null}
      </Alert.Content>
      <Button loading={isRetrying} size="sm" variant="outline" onClick={retry}>
        {t('common.retry')}
      </Button>
    </Alert.Root>
  );
};

/** The first control inside `container`, else the container itself (which must then take `tabIndex={-1}`). */
export const firstControlIn = (container: HTMLElement | null): HTMLElement | null =>
  container?.querySelector<HTMLElement>('a[href], button:not([disabled])') ?? container;
