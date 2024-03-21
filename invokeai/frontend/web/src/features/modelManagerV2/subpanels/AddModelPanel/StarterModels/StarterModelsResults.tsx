import { Flex, IconButton, Input, InputGroup, InputRightElement } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import type { ChangeEventHandler } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import type { GetStarterModelsResponse } from 'services/api/endpoints/models';

import { StarterModelsResultItem } from './StartModelsResultItem';

type StarterModelsResultsProps = {
  results: NonNullable<GetStarterModelsResponse>;
};

export const StarterModelsResults = ({ results }: StarterModelsResultsProps) => {
  const { t } = useTranslation();
  const [searchTerm, setSearchTerm] = useState('');

  const filteredResults = useMemo(() => {
    return results.filter((result) => {
      const name = result.name.toLowerCase();
      const type = result.type.toLowerCase();
      return name.includes(searchTerm.toLowerCase()) || type.includes(searchTerm.toLowerCase());
    });
  }, [results, searchTerm]);

  const handleSearch: ChangeEventHandler<HTMLInputElement> = useCallback((e) => {
    setSearchTerm(e.target.value.trim());
  }, []);

  const clearSearch = useCallback(() => {
    setSearchTerm('');
  }, []);

  return (
    <Flex flexDir="column" gap={3} height="100%">
      <Flex justifyContent="flex-end" alignItems="center">
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
      <Flex height="100%" layerStyle="third" borderRadius="base" p={3}>
        <ScrollableContent>
          <Flex flexDir="column" gap={3}>
            {filteredResults.map((result) => (
              <StarterModelsResultItem key={result.source} result={result} />
            ))}
          </Flex>
        </ScrollableContent>
      </Flex>
    </Flex>
  );
};
