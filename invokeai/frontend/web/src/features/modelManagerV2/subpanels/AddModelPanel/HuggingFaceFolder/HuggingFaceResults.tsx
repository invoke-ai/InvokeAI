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
import { useInstallModelMutation } from 'services/api/endpoints/models';

import { HuggingFaceResultItem } from './HuggingFaceResultItem';

type HuggingFaceResultsProps = {
  results: string[];
};

export const HuggingFaceResults = ({ results }: HuggingFaceResultsProps) => {
  const { t } = useTranslation();
  const [searchTerm, setSearchTerm] = useState('');
  const dispatch = useAppDispatch();

  const [installModel] = useInstallModelMutation();

  const filteredResults = useMemo(() => {
    return results.filter((result) => {
      const modelName = result.split('/').slice(-1)[0];
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
      installModel({ source: result })
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
  }, [filteredResults, installModel, dispatch, t]);

  return (
    <>
      <Divider mt={6} />
      <Flex flexDir="column" gap={2} mt={4} height="100%">
        <Flex justifyContent="space-between" alignItems="center">
          <Heading fontSize="md" as="h4">
            {t('modelManager.availableModels')}
          </Heading>
          <Flex alignItems="center" gap="4">
            <Button size="sm" onClick={handleAddAll} isDisabled={results.length === 0}>
              {t('modelManager.addAll')}
            </Button>
            <InputGroup maxW="300px" size="xs">
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
                  />
                </InputRightElement>
              )}
            </InputGroup>
          </Flex>
        </Flex>
        <Flex height="100%" layerStyle="third" borderRadius="base" p={4} mt={4} mb={4}>
          <ScrollableContent>
            <Flex flexDir="column" gap={3}>
              {filteredResults.map((result) => (
                <HuggingFaceResultItem key={result} result={result} />
              ))}
            </Flex>
          </ScrollableContent>
        </Flex>
      </Flex>
    </>
  );
};
