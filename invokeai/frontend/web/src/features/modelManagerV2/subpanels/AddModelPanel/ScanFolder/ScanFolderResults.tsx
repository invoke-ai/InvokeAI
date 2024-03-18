import {
  Button,
  Divider,
  Flex,
  Heading,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { ChangeEventHandler } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { type ScanFolderResponse, useInstallModelMutation } from 'services/api/endpoints/models';

import { ScanModelResultItem } from './ScanFolderResultItem';

type ScanModelResultsProps = {
  results: ScanFolderResponse;
};

export const ScanModelsResults = ({ results }: ScanModelResultsProps) => {
  const { t } = useTranslation();
  const [searchTerm, setSearchTerm] = useState('');
  const dispatch = useAppDispatch();

  const [installModel] = useInstallModelMutation();

  const filteredResults = useMemo(() => {
    return results.filter((result) => {
      const modelName = result.path.split('\\').slice(-1)[0];
      return modelName?.toLowerCase().includes(searchTerm.toLowerCase());
    });
  }, [results, searchTerm]);

  const handleSearch: ChangeEventHandler<HTMLInputElement> = useCallback((e) => {
    setSearchTerm(e.target.value.trim());
  }, []);

  const clearSearch = useCallback(() => {
    setSearchTerm('');
  }, []);

  const handleAddAll = useCallback(() => {
    for (const result of filteredResults) {
      if (result.is_installed) {
        continue;
      }
      installModel({ source: result.path })
        .unwrap()
        .then((_) => {
          dispatch(
            addToast(
              makeToast({
                title: t('toast.modelAddedSimple'),
                status: 'success',
              })
            )
          );
        })
        .catch((error) => {
          if (error) {
            dispatch(
              addToast(
                makeToast({
                  title: `${error.data.detail} `,
                  status: 'error',
                })
              )
            );
          }
        });
    }
  }, [installModel, filteredResults, dispatch, t]);

  return (
    <>
      <Divider />
      <Flex flexDir="column" gap={3} height="100%">
        <Flex justifyContent="space-between" alignItems="center">
          <Heading size="sm">{t('modelManager.scanResults')}</Heading>
          <Flex alignItems="center" gap={3}>
            <Button size="sm" onClick={handleAddAll} isDisabled={filteredResults.length === 0}>
              {t('modelManager.installAll')}
            </Button>
            <InputGroup w={64} size="xs">
              <Input
                placeholder={t('modelManager.search')}
                value={searchTerm}
                data-testid="board-search-input"
                onChange={handleSearch}
                size="xs"
              />

              {searchTerm && (
                <InputRightElement h="full" pe={2}>
                  <IconButton
                    size="sm"
                    variant="link"
                    aria-label={t('boards.clearSearch')}
                    icon={<PiXBold />}
                    onClick={clearSearch}
                    flexShrink={0}
                  />
                </InputRightElement>
              )}
            </InputGroup>
          </Flex>
        </Flex>
        <Flex height="100%" layerStyle="third" borderRadius="base" p={3}>
          <ScrollableContent>
            <Flex flexDir="column" gap={3}>
              {filteredResults.map((result) => (
                <ScanModelResultItem key={result.path} result={result} />
              ))}
            </Flex>
          </ScrollableContent>
        </Flex>
      </Flex>
    </>
  );
};
