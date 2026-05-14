import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  Button,
  Combobox,
  Flex,
  FormControl,
  FormLabel,
  Image,
  Input,
  Radio,
  RadioGroup,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasProjectSave } from 'features/controlLayers/hooks/useCanvasProjectSave';
import { $currentCanvasProjectName } from 'features/controlLayers/store/currentCanvasProject';
import { atom } from 'nanostores';
import type { ChangeEvent, RefObject } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { useGetCanvasProjectDTOQuery } from 'services/api/endpoints/canvasProjects';

const $isOpen = atom(false);

type SaveDestination = 'server' | 'file';
type SaveMode = 'create' | 'update';

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
  const { saveCanvasProjectAsFile, saveCanvasProjectToServer, updateCanvasProjectOnServer } = useCanvasProjectSave();
  const currentProjectName = useStore($currentCanvasProjectName);

  // Pre-fill name/board from the currently-loaded project when we know it. Skip the query
  // entirely if there's nothing loaded so we don't make a wasted request.
  const { data: currentProject } = useGetCanvasProjectDTOQuery(currentProjectName ?? '', {
    skip: !currentProjectName,
  });

  const [name, setName] = useState('Canvas Project');
  const [destination, setDestination] = useState<SaveDestination>('server');
  const [mode, setMode] = useState<SaveMode>(currentProjectName ? 'update' : 'create');
  const [boardId, setBoardId] = useState<string>('none');

  // When the loaded project's DTO arrives, hydrate the form. Doing this in an effect avoids a
  // jarring re-render-during-render and lets the user override the values afterwards.
  useEffect(() => {
    if (currentProject) {
      setName(currentProject.name);
      setBoardId(currentProject.board_id ?? 'none');
    }
  }, [currentProject]);

  const onChangeName = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value);
  }, []);

  const onChangeDestination = useCallback((v: string) => {
    setDestination(v === 'file' ? 'file' : 'server');
  }, []);

  const onChangeMode = useCallback((v: string) => {
    setMode(v === 'update' ? 'update' : 'create');
  }, []);

  const { boardOptions } = useListAllBoardsQuery(
    {},
    {
      selectFromResult: ({ data }) => {
        const options: ComboboxOption[] = [{ label: t('common.none'), value: 'none' }].concat(
          (data ?? []).map(({ board_id, board_name }) => ({
            label: board_name,
            value: board_id,
          }))
        );
        return { boardOptions: options };
      },
    }
  );

  const boardValue = useMemo(() => boardOptions.find((o) => o.value === boardId), [boardOptions, boardId]);

  const onChangeBoard = useCallback<ComboboxOnChange>((v) => {
    if (!v) {
      return;
    }
    setBoardId(v.value);
  }, []);

  const onClose = useCallback(() => {
    $isOpen.set(false);
  }, []);

  const onSave = useCallback(() => {
    if (destination === 'server') {
      if (mode === 'update' && currentProjectName) {
        void updateCanvasProjectOnServer(currentProjectName, name);
      } else {
        void saveCanvasProjectToServer(name, boardId === 'none' ? undefined : boardId);
      }
    } else {
      void saveCanvasProjectAsFile(name);
    }
    $isOpen.set(false);
  }, [
    destination,
    mode,
    currentProjectName,
    name,
    boardId,
    saveCanvasProjectAsFile,
    saveCanvasProjectToServer,
    updateCanvasProjectOnServer,
  ]);

  return (
    <AlertDialogContent>
      <AlertDialogHeader fontSize="lg" fontWeight="bold">
        {t('controlLayers.canvasProject.saveProject')}
      </AlertDialogHeader>

      <AlertDialogBody>
        <Flex flexDir="column" gap="4">
          <FormControl alignItems="flex-start">
            <FormLabel mt="2">{t('controlLayers.canvasProject.projectName')}</FormLabel>
            <Input value={name} onChange={onChangeName} placeholder={t('controlLayers.canvasProject.projectName')} />
          </FormControl>

          <FormControl alignItems="flex-start">
            <FormLabel mt="2">{t('controlLayers.canvasProject.saveDestination')}</FormLabel>
            <RadioGroup value={destination} onChange={onChangeDestination} w="full">
              <Flex flexDir="column" gap="2">
                <Radio value="server">{t('controlLayers.stagingArea.saveToGallery')}</Radio>
                <Radio value="file">{t('controlLayers.canvasProject.downloadAsFile')}</Radio>
              </Flex>
            </RadioGroup>
          </FormControl>

          {destination === 'server' && currentProjectName && (
            <FormControl alignItems="flex-start">
              <RadioGroup value={mode} onChange={onChangeMode} w="full">
                <Flex flexDir="column" gap="3">
                  <Radio value="update" alignItems="flex-start">
                    <Flex flexDir="column" gap="1" ml="1">
                      <Text>{t('controlLayers.canvasProject.updateExisting')}</Text>
                      {currentProject && (
                        <Flex alignItems="center" gap="2">
                          {currentProject.thumbnail_url && (
                            <Image
                              src={currentProject.thumbnail_url}
                              alt=""
                              boxSize="40px"
                              objectFit="cover"
                              borderRadius="base"
                              flexShrink={0}
                            />
                          )}
                          <Flex flexDir="column">
                            <Text fontSize="sm" fontWeight="semibold" noOfLines={1}>
                              {currentProject.name}
                            </Text>
                            <Text fontSize="xs" color="base.300" noOfLines={1}>
                              {currentProject.board_id
                                ? t('controlLayers.canvasProject.board')
                                : t('common.none')}
                              {' · '}
                              {currentProject.width}×{currentProject.height}
                            </Text>
                          </Flex>
                        </Flex>
                      )}
                    </Flex>
                  </Radio>
                  <Radio value="create">{t('controlLayers.canvasProject.createNew')}</Radio>
                </Flex>
              </RadioGroup>
            </FormControl>
          )}

          {destination === 'server' && mode === 'create' && (
            <FormControl alignItems="flex-start">
              <FormLabel mt="2">{t('controlLayers.canvasProject.board')}</FormLabel>
              <Combobox
                value={boardValue}
                options={boardOptions}
                onChange={onChangeBoard}
                placeholder={t('boards.selectBoard')}
              />
            </FormControl>
          )}
        </Flex>
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
