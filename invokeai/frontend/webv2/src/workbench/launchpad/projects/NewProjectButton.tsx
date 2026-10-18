import { Icon } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { toaster } from '@platform/ui/toaster';
import { PlusIcon } from 'lucide-react';
import { lazy, Suspense, useCallback, useState, type ComponentProps, type ComponentType } from 'react';
import { useTranslation } from 'react-i18next';

import type { NewProjectDialog } from './NewProjectDialog';

/**
 * Opens the New-project dialog; the Home tiles stay the one-click path. The dialog pulls the generation policies and
 * model catalog, so its module loads on the first open rather than with the Launchpad.
 */

const loadNewProjectDialog = () => import('./NewProjectDialog');
// A chunk that fails to load (an offline tab, a stale deploy) renders nothing; the open handler closes and says so.
type NewProjectDialogComponent = ComponentType<ComponentProps<typeof NewProjectDialog>>;
const NothingToShow: NewProjectDialogComponent = () => null;
const LazyNewProjectDialog = lazy<NewProjectDialogComponent>(() =>
  loadNewProjectDialog()
    .then((module) => ({ default: module.NewProjectDialog as NewProjectDialogComponent }))
    .catch(() => ({ default: NothingToShow }))
);
// Modal from the click onward: while the dialog's module is still on its way, shortcuts must already be off.
const PENDING_DIALOG = <Dialog.Pending />;

export const NewProjectButton = ({ variant = 'solid' }: { variant?: 'outline' | 'solid' }) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  // Once mounted the dialog stays, so its close animation and the lazy chunk are paid for once.
  const [hasOpened, setHasOpened] = useState(false);

  const open = useCallback(() => {
    setHasOpened(true);
    setIsOpen(true);
    // Start the catalog loads alongside the chunk, so the form rarely shows its loading state.
    void loadNewProjectDialog()
      .then((module) => module.prepareNewProjectDialog())
      .catch(() => {
        setIsOpen(false);
        toaster.create({ title: t('projects.newProjectDialog.loadFailed'), type: 'error' });
      });
  }, [t]);
  const close = useCallback(() => setIsOpen(false), []);

  return (
    <>
      <Button aria-haspopup="dialog" variant={variant} onClick={open}>
        <Icon as={PlusIcon} boxSize="3.5" />
        {t('projects.newProject')}
      </Button>
      {hasOpened ? (
        <Suspense fallback={isOpen ? PENDING_DIALOG : null}>
          <LazyNewProjectDialog isOpen={isOpen} onClose={close} />
        </Suspense>
      ) : null}
    </>
  );
};
