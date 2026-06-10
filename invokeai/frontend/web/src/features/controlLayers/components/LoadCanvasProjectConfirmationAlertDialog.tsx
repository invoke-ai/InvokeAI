import { ConfirmationAlertDialog, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasProjectLoad } from 'features/controlLayers/hooks/useCanvasProjectLoad';
import { CANVAS_PROJECT_EXTENSION } from 'features/controlLayers/util/canvasProjectFile';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type PendingLoad = { kind: 'file'; file: File } | { kind: 'server'; projectName: string };

const $pending = atom<PendingLoad | null>(null);

const openFileDialog = (onFileSelected: (file: File) => void) => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = CANVAS_PROJECT_EXTENSION;
  input.onchange = () => {
    const file = input.files?.[0];
    if (file) {
      onFileSelected(file);
    }
  };
  input.click();
};

/**
 * Opens the OS file picker, then queues the resulting file for the load-confirmation dialog.
 * Used by the File menu / context menu "Load Canvas Project from File" entry.
 */
export const useLoadCanvasProjectFromFileWithDialog = () => {
  return useCallback(() => {
    openFileDialog((file) => {
      $pending.set({ kind: 'file', file });
    });
  }, []);
};

/**
 * Queues a server-stored canvas project for the load-confirmation dialog. The ZIP will be
 * fetched from `/api/v1/canvas_projects/i/{name}/full` on accept.
 *
 * Returns a stable callback so consumers (gallery viewer toolbar, click-to-load actions) can
 * pass it as an onClick handler.
 */
export const useLoadCanvasProjectFromServerWithDialog = () => {
  return useCallback((projectName: string) => {
    $pending.set({ kind: 'server', projectName });
  }, []);
};

// Kept for backwards compatibility with the existing context-menu wiring.
export const useLoadCanvasProjectWithDialog = useLoadCanvasProjectFromFileWithDialog;

export const LoadCanvasProjectConfirmationAlertDialog = memo(() => {
  useAssertSingleton('LoadCanvasProjectConfirmationAlertDialog');
  const { t } = useTranslation();
  const { loadCanvasProjectFromFile, loadCanvasProjectFromServer } = useCanvasProjectLoad();
  const pending = useStore($pending);

  const onClose = useCallback(() => {
    $pending.set(null);
  }, []);

  const onAccept = useCallback(() => {
    const p = $pending.get();
    if (p?.kind === 'file') {
      void loadCanvasProjectFromFile(p.file);
    } else if (p?.kind === 'server') {
      void loadCanvasProjectFromServer(p.projectName);
    }
    $pending.set(null);
  }, [loadCanvasProjectFromFile, loadCanvasProjectFromServer]);

  return (
    <ConfirmationAlertDialog
      isOpen={pending !== null}
      onClose={onClose}
      title={t('controlLayers.canvasProject.loadProject')}
      acceptCallback={onAccept}
      acceptButtonText={t('common.load')}
      useInert={false}
    >
      <Flex flexDir="column" gap={2}>
        <Text>{t('controlLayers.canvasProject.loadWarning')}</Text>
      </Flex>
    </ConfirmationAlertDialog>
  );
});

LoadCanvasProjectConfirmationAlertDialog.displayName = 'LoadCanvasProjectConfirmationAlertDialog';
