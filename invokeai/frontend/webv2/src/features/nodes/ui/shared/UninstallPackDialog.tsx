/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import type { NodePackInfo } from '@features/nodes/core/catalog';

import { getPackWorkflowCount } from '@features/nodes/data/api';
import { useNodePackActions } from '@features/nodes/ui/shared/useNodePackActions';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { ConfirmDialog } from '@platform/ui';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Warn that uninstall deletes imported workflows; use a best-effort tag count and generic copy if unavailable or
 * zero.
 */
export const UninstallPackDialog = ({
  finalFocusEl,
  onClose,
  onUninstalled,
  pack,
}: {
  /** Focus destination on close; list rows resolve it late because the uninstall removes the opener. */
  finalFocusEl?: () => HTMLElement | null;
  onClose: () => void;
  onUninstalled?: (packName: string) => void;
  pack: NodePackInfo | null;
}) => {
  const { t } = useTranslation();
  const { uninstall } = useNodePackActions();
  // Keyed by pack name so a count fetched for one pack never leaks into
  // another's dialog; no reset needed when the target changes.
  const [countState, setCountState] = useState<{ count: number; packName: string } | null>(null);

  useEffect(() => {
    if (pack === null) {
      return;
    }

    const owner = captureAccountScope();

    getPackWorkflowCount(pack.name, owner.signal)
      .then((count) => {
        if (isAccountScopeCurrent(owner)) {
          setCountState({ count, packName: pack.name });
        }
      })
      .catch(() => undefined);
  }, [pack]);

  const workflowCount = countState !== null && countState.packName === pack?.name ? countState.count : 0;

  return (
    <ConfirmDialog
      body={
        workflowCount > 0 ? t('nodes.uninstallBodyWithWorkflows', { count: workflowCount }) : t('nodes.uninstallBody')
      }
      confirmLabel={t('nodes.uninstallPack')}
      finalFocusEl={finalFocusEl}
      isOpen={pack !== null}
      title={t('nodes.uninstallTitle', { name: pack?.name ?? t('nodes.nodePack') })}
      onClose={onClose}
      onConfirm={async () => {
        if (pack) {
          await uninstall(pack, onUninstalled);
          onClose();
        }
      }}
    />
  );
};
