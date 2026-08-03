import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { SettingsIcon } from 'lucide-react';
import { useCallback } from 'react';

import { SettingsDialogHost } from './SettingsDialogHost';
import { openWorkbenchSettings } from './settingsDialogStore';

/**
 * The Launchpad's settings entry point, kept deliberately thin. The workbench
 * shell opens settings from its app menu instead and mounts
 * {@link SettingsDialogHost} on its own.
 */
export const SettingsButton = () => {
  const handleOpen = useCallback(() => openWorkbenchSettings(), []);

  return (
    <>
      <Tooltip content="Settings">
        <IconButton aria-label="Settings" size="sm" variant="ghost" onClick={handleOpen}>
          <SettingsIcon />
        </IconButton>
      </Tooltip>
      <SettingsDialogHost />
    </>
  );
};
