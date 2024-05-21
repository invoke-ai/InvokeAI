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
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { toast, ToastID } from 'features/toast/toast';
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
          toast({
            id: ToastID.MODEL_INSTALL_QUEUED,
            title: t('toast.modelAddedSimple'),
            status: 'success',
          });
        })
        .catch((error) => {
          if (error) {
            toast({
              id: ToastID.MODEL_INSTALL_QUEUE_FAILED,
              title: `${error.data.detail} `,
              status: 'error',
            });
          }
        });
    }
  }, [filteredResults, installModel, t]);

  return (
    <>
      <Divider />
      <Flex flexDir="column" gap={3} height="100%">
        <Flex justifyContent="space-between" alignItems="center">
          <Heading size="sm">{t('modelManager.availableModels')}</Heading>
          <Flex alignItems="center" gap={3}>
            <Button size="sm" onClick={handleAddAll} isDisabled={results.length === 0} flexShrink={0}>
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
                <HuggingFaceResultItem key={result} result={result} />
              ))}
            </Flex>
          </ScrollableContent>
        </Flex>
      </Flex>
    </>
  );
};
