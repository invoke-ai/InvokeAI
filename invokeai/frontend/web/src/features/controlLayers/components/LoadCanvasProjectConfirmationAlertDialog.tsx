import { ConfirmationAlertDialog, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasProjectLoad } from 'features/controlLayers/hooks/useCanvasProjectLoad';
import { CANVAS_PROJECT_EXTENSION } from 'features/controlLayers/util/canvasProjectFile';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const $pendingFile = atom<File | null>(null);

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

export const useLoadCanvasProjectWithDialog = () => {
  const openDialog = useCallback(() => {
    openFileDialog((file) => {
      $pendingFile.set(file);
    });
  }, []);

  return openDialog;
};

export const LoadCanvasProjectConfirmationAlertDialog = memo(() => {
  useAssertSingleton('LoadCanvasProjectConfirmationAlertDialog');
  const { t } = useTranslation();
  const { loadCanvasProject } = useCanvasProjectLoad();
  const pendingFile = useStore($pendingFile);

  const onClose = useCallback(() => {
    $pendingFile.set(null);
  }, []);

  const onAccept = useCallback(() => {
    const file = $pendingFile.get();
    if (file) {
      void loadCanvasProject(file);
    }
    $pendingFile.set(null);
  }, [loadCanvasProject]);

  return (
    <ConfirmationAlertDialog
      isOpen={pendingFile !== null}
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
