import {
  Text,
  Flex,
  Heading,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Divider,
  Box,
} from '@invoke-ai/ui-library';
import { t } from 'i18next';
import { ChangeEventHandler, useCallback, useMemo, useState } from 'react';
import { PiXBold } from 'react-icons/pi';
import ScrollableContent from '../../../../../common/components/OverlayScrollbars/ScrollableContent';

export const ScanModelsResults = ({ results }: { results: string[] }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredResults, setFilteredResults] = useState(results);

  const handleSearch: ChangeEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      setSearchTerm(e.target.value);
      setFilteredResults(
        results.filter((result) => {
          const modelName = result.split('\\').slice(-1)[0];
          return modelName?.includes(e.target.value);
        })
      );
    },
    [results]
  );

  const clearSearch = useCallback(() => {
    setSearchTerm('');
  }, []);

  return (
    <>
      <Divider mt={4} />
      <Flex flexDir="column" gap={2} mt={4} height="100%">
        <Flex justifyContent="space-between" alignItems="center">
          <Heading fontSize="md" as="h4">
            Scan Results
          </Heading>
          <InputGroup maxW="300px" size="xs">
            <Input
              placeholder={t('modelManager.search')}
              value={searchTerm || ''}
              data-testid="board-search-input"
              onChange={handleSearch}
              size="xs"
            />

            {!!searchTerm?.length && (
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
        <Flex height="100%" layerStyle="third" borderRadius="base" p={4} mt={4} mb={4}>
          <ScrollableContent>
            {filteredResults.map((result) => (
              <Flex key={result} fontSize="sm" flexDir="column">
                <Text fontWeight="semibold">{result.split('\\').slice(-1)[0]}</Text>
                <Text variant="subtext">{result}</Text>
              </Flex>
            ))}
          </ScrollableContent>
        </Flex>
      </Flex>
    </>
  );
};
