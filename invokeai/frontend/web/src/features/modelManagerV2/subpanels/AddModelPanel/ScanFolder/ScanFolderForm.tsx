import { Button, Flex, FormControl, FormErrorMessage, FormLabel, Input } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setScanPath } from 'features/modelManagerV2/store/modelManagerV2Slice';
import type { ChangeEventHandler } from 'react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyScanFolderQuery } from 'services/api/endpoints/models';

import { ScanModelsResults } from './ScanFolderResults';

export const ScanModelsForm = () => {
  const scanPath = useAppSelector((state) => state.modelmanagerV2.scanPath);
  const dispatch = useAppDispatch();
  const [errorMessage, setErrorMessage] = useState('');
  const { t } = useTranslation();

  const [_scanFolder, { isLoading, data }] = useLazyScanFolderQuery();

  const scanFolder = useCallback(async () => {
    _scanFolder({ scan_path: scanPath }).catch((error) => {
      if (error) {
        setErrorMessage(error.data.detail);
      }
    });
  }, [_scanFolder, scanPath]);

  const handleSetScanPath: ChangeEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      dispatch(setScanPath(e.target.value));
      setErrorMessage('');
    },
    [dispatch]
  );

  return (
    <Flex flexDir="column" height="100%">
      <FormControl isInvalid={!!errorMessage.length} w="full">
        <Flex flexDir="column" w="full">
          <Flex gap={2} alignItems="flex-end" justifyContent="space-between">
            <Flex direction="column" w="full">
              <FormLabel>{t('common.folder')}</FormLabel>
              <Input value={scanPath} onChange={handleSetScanPath} />
            </Flex>

            <Button
              onClick={scanFolder}
              isLoading={isLoading}
              isDisabled={scanPath === undefined || scanPath.length === 0}
            >
              {t('modelManager.scanFolder')}
            </Button>
          </Flex>
          {!!errorMessage.length && <FormErrorMessage>{errorMessage}</FormErrorMessage>}
        </Flex>
      </FormControl>
      {data && <ScanModelsResults results={data} />}
    </Flex>
  );
};
