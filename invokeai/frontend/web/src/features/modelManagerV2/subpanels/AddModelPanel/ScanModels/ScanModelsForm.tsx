import { Flex, FormControl, FormLabel, Input, Button, FormErrorMessage, Divider } from '@invoke-ai/ui-library';
import { ChangeEventHandler, useCallback, useState } from 'react';
import { useLazyScanModelsQuery } from '../../../../../services/api/endpoints/models';
import { useTranslation } from 'react-i18next';
import { ScanModelsResults } from './ScanModelsResults';

export const ScanModelsForm = () => {
  const [scanPath, setScanPath] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [results, setResults] = useState<string[] | undefined>();
  const { t } = useTranslation();

  const [_scanModels, { isLoading }] = useLazyScanModelsQuery();

  const handleSubmitScan = useCallback(async () => {
    _scanModels({ scan_path: scanPath })
      .unwrap()
      .then((result) => {
        setResults(result);
      })
      .catch((error) => {
        if (error) {
          setErrorMessage(error.data.detail);
        }
      });
  }, [scanPath]);

  const handleSetScanPath: ChangeEventHandler<HTMLInputElement> = useCallback((e) => {
    setScanPath(e.target.value);
    setErrorMessage('');
  }, []);

  return (
    <>
      <FormControl isInvalid={!!errorMessage.length} w="full">
        <Flex flexDir="column" w="full">
          <Flex gap={2} alignItems="flex-end" justifyContent="space-between">
            <Flex direction="column" w="full">
              <FormLabel>{t('common.folder')}</FormLabel>
              <Input value={scanPath} onChange={handleSetScanPath} />
            </Flex>

            <Button onClick={handleSubmitScan} isLoading={isLoading} isDisabled={scanPath.length === 0}>
              {t('modelManager.scanFolder')}
            </Button>
          </Flex>
          {!!errorMessage.length && <FormErrorMessage>{errorMessage}</FormErrorMessage>}
        </Flex>
      </FormControl>

      {results && <ScanModelsResults results={results} />}
    </>
  );
};
