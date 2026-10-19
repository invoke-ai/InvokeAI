/* eslint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ListContextMenuAnchor } from '@platform/ui/list/ListItem';

import { Menu, Portal } from '@chakra-ui/react';
import { useActiveInstallSources } from '@features/models/data/installsStore';
import { openInstallQueue, openModelDetail } from '@features/models/ui/uiStore';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { CheckIcon, DownloadIcon, ListIcon, SquareArrowOutUpRightIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** What a list knows about one installable source; active install jobs are read live by its controls. */
export interface InstallSourceStatus {
  /** The library model this source landed as; implies installed. */
  installedModelKey: string | null;
  /** Already in the library, without a known model to open. */
  isInstalled: boolean;
  /** The install POST is in flight (before a job exists). */
  isPending: boolean;
}

/** The next step a source offers; the row's button and its menu both follow it. */
export type InstallSourceStage =
  | { kind: 'installed'; modelKey: string | null }
  | { kind: 'installing' }
  | { kind: 'available' };

export const useInstallSourceStage = (
  source: string,
  { installedModelKey, isInstalled, isPending }: InstallSourceStatus
): InstallSourceStage => {
  const activeSources = useActiveInstallSources();

  if (isInstalled || installedModelKey !== null) {
    return { kind: 'installed', modelKey: installedModelKey };
  }

  return isPending || activeSources.has(source) ? { kind: 'installing' } : { kind: 'available' };
};

export interface InstallSourceMenuTarget extends ListContextMenuAnchor {
  source: string;
}

/**
 * One context menu per list: `target` is the row that was right-clicked, until the menu closes; `shown` keeps it
 * through the menu's exit animation. `open` keeps its identity so memoized rows skip re-rendering when only the
 * target changes.
 */
export const useInstallSourceMenu = () => {
  const [target, setTarget] = useState<InstallSourceMenuTarget | null>(null);
  const { release, value: shown } = useExitRetainedValue(target);
  const open = useCallback((source: string, anchor: ListContextMenuAnchor) => setTarget({ ...anchor, source }), []);

  return {
    close: () => {
      target?.restoreFocus();
      setTarget(null);
    },
    open,
    release,
    shown,
    target,
  };
};

/**
 * The list's single row menu. The list resolves `status` for the target on every render, so the menu follows the
 * source while it is open; it offers the same next step as the row's install button.
 */
export const InstallSourceContextMenu = ({
  isOpen,
  onClose,
  onExitComplete,
  onInstall,
  status,
  target,
}: {
  isOpen: boolean;
  onClose: () => void;
  /** Releases the retained target once the menu has animated out. */
  onExitComplete: () => void;
  /** Installs the target source. */
  onInstall: () => void;
  /** Null when the target is no longer listed. */
  status: InstallSourceStatus | null;
  /** The shown target, kept while the menu animates out. */
  target: InstallSourceMenuTarget | null;
}) => (
  <Menu.Root
    key={target ? target.source : 'none'}
    lazyMount
    open={isOpen}
    positioning={{
      getAnchorRect: () => (target ? { height: 1, width: 1, x: target.x, y: target.y } : null),
      placement: 'bottom-start',
    }}
    unmountOnExit
    onExitComplete={onExitComplete}
    onOpenChange={(event) => {
      if (!event.open) {
        onClose();
      }
    }}
  >
    <Portal>
      <Menu.Positioner>
        {target && status ? (
          <MenuContent minW="10rem">
            <InstallSourceMenuItem source={target.source} status={status} onInstall={onInstall} />
          </MenuContent>
        ) : null}
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
);

/** Mounted only while the menu is open, so the list holds no install-job subscription for it otherwise. */
const InstallSourceMenuItem = ({
  onInstall,
  source,
  status,
}: {
  onInstall: () => void;
  source: string;
  status: InstallSourceStatus;
}) => {
  const { t } = useTranslation();
  const stage = useInstallSourceStage(source, status);

  if (stage.kind === 'installed' && stage.modelKey !== null) {
    const { modelKey } = stage;

    return (
      <MenuActionItem
        icon={SquareArrowOutUpRightIcon}
        label={t('models.viewModel')}
        value="view-model"
        onSelect={() => openModelDetail(modelKey)}
      />
    );
  }

  if (stage.kind === 'installed') {
    // A status rather than a dead end: the row's own control is the same badge, and keeping a menu keeps right-click
    // consistent across rows (no browser menu on some of them) and across an install finishing while it is open.
    return <MenuActionItem disabled icon={CheckIcon} label={t('models.installed')} value="installed" onSelect={noop} />;
  }

  if (stage.kind === 'installing') {
    return (
      <MenuActionItem icon={ListIcon} label={t('models.viewQueue')} value="view-queue" onSelect={openInstallQueue} />
    );
  }

  return <MenuActionItem icon={DownloadIcon} label={t('models.install')} value="install" onSelect={onInstall} />;
};

const noop = () => undefined;
