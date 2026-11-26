import type { GalleryArchive } from '@features/gallery/archiveRecovery';
import type { AuthSession } from '@features/identity';

import { recoverGalleryArchives } from '@features/gallery/archiveRecovery';
import { getAuthSession, subscribeAuthSession } from '@features/identity';
import { downloadBlob } from '@platform/browser/downloadBlob';
import { type AccountScope, captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { toaster } from '@platform/ui/toaster';
import i18next from 'i18next';

interface SessionSource {
  getSnapshot: () => AuthSession;
  subscribe: (listener: () => void) => () => void;
}

/** Calls `start` once per account lifetime that can use the app, with that lifetime's scope. */
export const watchAccountActivations = (session: SessionSource, start: (owner: AccountScope) => void): (() => void) => {
  let startedEpoch: number | null = null;
  const check = (): void => {
    const { accountEpoch, multiuserEnabled, phase, switchingAccount, user } = session.getSnapshot();
    const owner = captureAccountScope();

    if (
      phase !== 'ready' ||
      switchingAccount ||
      (multiuserEnabled && user === null) ||
      accountEpoch === startedEpoch ||
      owner.epoch !== accountEpoch
    ) {
      return;
    }
    startedEpoch = accountEpoch;
    start(owner);
  };

  const unsubscribe = session.subscribe(check);
  check();
  return unsubscribe;
};

const saveRecoveredArchive =
  (owner: AccountScope) =>
  ({ blob, fileName }: GalleryArchive): void => {
    if (!isAccountScopeCurrent(owner)) {
      return;
    }
    void downloadBlob(blob, fileName);
    toaster.create({ description: fileName, title: i18next.t('widgets.gallery.recoveredDownload'), type: 'success' });
  };

/** Saves archives prepared for an account while no tab was waiting for them, such as across a sign-in or reload. */
export const startArchiveRecovery = (): (() => void) =>
  watchAccountActivations({ getSnapshot: getAuthSession, subscribe: subscribeAuthSession }, (owner) => {
    void recoverGalleryArchives(owner, saveRecoveredArchive(owner));
  });
