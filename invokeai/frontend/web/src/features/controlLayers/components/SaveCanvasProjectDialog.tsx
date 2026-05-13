import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  Button,
  Flex,
  FormControl,
  FormLabel,
  Input,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasProjectSave } from 'features/controlLayers/hooks/useCanvasProjectSave';
import { atom } from 'nanostores';
import type { ChangeEvent, RefObject } from 'react';
import { memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const $isOpen = atom(false);

export const useSaveCanvasProjectWithDialog = () => {
  return useCallback(() => {
    $isOpen.set(true);
  }, []);
};

export const SaveCanvasProjectDialog = memo(() => {
  useAssertSingleton('SaveCanvasProjectDialog');
  const isOpen = useStore($isOpen);
  const cancelRef = useRef<HTMLButtonElement>(null);

  const onClose = useCallback(() => {
    $isOpen.set(false);
  }, []);

  return (
    <AlertDialog isOpen={isOpen} onClose={onClose} leastDestructiveRef={cancelRef} isCentered>
      {isOpen && <Content cancelRef={cancelRef} />}
    </AlertDialog>
  );
});

SaveCanvasProjectDialog.displayName = 'SaveCanvasProjectDialog';

const Content = memo(({ cancelRef }: { cancelRef: RefObject<HTMLButtonElement> }) => {
  const { t } = useTranslation();
  const { saveCanvasProject } = useCanvasProjectSave();
  const [name, setName] = useState('Canvas Project');

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value);
  }, []);

  const onClose = useCallback(() => {
    $isOpen.set(false);
  }, []);

  const onSave = useCallback(() => {
    void saveCanvasProject(name);
    $isOpen.set(false);
  }, [name, saveCanvasProject]);

  return (
    <AlertDialogContent>
      <AlertDialogHeader fontSize="lg" fontWeight="bold">
        {t('controlLayers.canvasProject.saveProject')}
      </AlertDialogHeader>

      <AlertDialogBody>
        <FormControl alignItems="flex-start">
          <FormLabel mt="2">{t('controlLayers.canvasProject.projectName')}</FormLabel>
          <Flex flexDir="column" width="full" gap="2">
            <Input value={name} onChange={onChange} placeholder={t('controlLayers.canvasProject.projectName')} />
          </Flex>
        </FormControl>
      </AlertDialogBody>

      <AlertDialogFooter>
        <Button ref={cancelRef} onClick={onClose}>
          {t('common.cancel')}
        </Button>
        <Button colorScheme="invokeBlue" onClick={onSave} ml={3} isDisabled={!name.trim()}>
          {t('common.save')}
        </Button>
      </AlertDialogFooter>
    </AlertDialogContent>
  );
});

Content.displayName = 'SaveCanvasProjectDialogContent';
