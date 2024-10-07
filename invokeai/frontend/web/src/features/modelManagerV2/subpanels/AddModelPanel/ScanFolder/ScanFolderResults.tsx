import {
  Button,
  Checkbox,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
} from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import type { ChangeEvent, ChangeEventHandler } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import type { ScanFolderResponse } from 'services/api/endpoints/models';

import { ScanModelResultItem } from './ScanFolderResultItem';

type ScanModelResultsProps = {
  results: ScanFolderResponse;
};

export const ScanModelsResults = memo(({ results }: ScanModelResultsProps) => {
  const { t } = useTranslation();
  const [searchTerm, setSearchTerm] = useState('');
  const [inplace, setInplace] = useState(true);
  const [installModel] = useInstallModel();

  const filteredResults = useMemo(() => {
    return results.filter((result) => {
      const modelName = result.path.split('\\').slice(-1)[0];
      return modelName?.toLowerCase().includes(searchTerm.toLowerCase());
    });
  }, [results, searchTerm]);

  const handleSearch: ChangeEventHandler<HTMLInputElement> = useCallback((e) => {
    setSearchTerm(e.target.value.trim());
  }, []);

  const onChangeInplace = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setInplace(e.target.checked);
  }, []);

  const clearSearch = useCallback(() => {
    setSearchTerm('');
  }, []);

  const handleAddAll = useCallback(() => {
    for (const result of filteredResults) {
      if (result.is_installed) {
        continue;
      }
      installModel({ source: result.path, inplace });
    }
  }, [filteredResults, installModel, inplace]);

  const handleInstallOne = useCallback(
    (source: string) => {
      installModel({ source, inplace });
    },
    [installModel, inplace]
  );

  return (
    <>
      <Divider />
      <Flex flexDir="column" gap={3} height="100%">
        <Flex justifyContent="space-between" alignItems="center">
          <Heading size="sm">{t('modelManager.scanResults')}</Heading>
          <Flex alignItems="center" gap={3}>
            <FormControl w="min-content">
              <FormLabel m={0}>{t('modelManager.inplaceInstall')}</FormLabel>
              <Checkbox isChecked={inplace} onChange={onChangeInplace} size="md" />
            </FormControl>
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
                <ScanModelResultItem key={result.path} result={result} installModel={handleInstallOne} />
              ))}
            </Flex>
          </ScrollableContent>
        </Flex>
      </Flex>
    </>
  );
});

ScanModelsResults.displayName = 'ScanModelsResults';
