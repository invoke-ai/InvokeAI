import {
  Box,
  Flex,
  Icon,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Text,
  Tooltip,
} from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { map, size } from 'lodash-es';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiInfoBold, PiXBold } from 'react-icons/pi';
import type { GetStarterModelsResponse } from 'services/api/endpoints/models';

import { StarterBundle } from './StarterBundle';
import { StarterModelsResultItem } from './StarterModelsResultItem';

type StarterModelsResultsProps = {
  results: NonNullable<GetStarterModelsResponse>;
};

export const StarterModelsResults = memo(({ results }: StarterModelsResultsProps) => {
  const { t } = useTranslation();
  const [searchTerm, setSearchTerm] = useState('');

  const filteredResults = useMemo(() => {
    return results.starter_models.filter((result) => {
      const trimmedSearchTerm = searchTerm.trim().toLowerCase();
      const matchStrings = [
        result.name.toLowerCase(),
        result.type.toLowerCase().replaceAll('_', ' '),
        result.description.toLowerCase(),
      ];
      if (result.type === 'spandrel_image_to_image') {
        matchStrings.push('upscale');
        matchStrings.push('post-processing');
        matchStrings.push('postprocessing');
        matchStrings.push('post processing');
      }
      return matchStrings.some((matchString) => matchString.includes(trimmedSearchTerm));
    });
  }, [results, searchTerm]);

  const handleSearch: ChangeEventHandler<HTMLInputElement> = useCallback((e) => {
    setSearchTerm(e.target.value);
  }, []);

  const clearSearch = useCallback(() => {
    setSearchTerm('');
  }, []);

  return (
    <Flex flexDir="column" gap={3} height="100%">
      <Flex justifyContent="space-between" alignItems="center">
        {size(results.starter_bundles) > 0 && (
          <Flex gap={4} alignItems="center">
            <Flex gap={1} alignItems="center">
              <Text color="base.200" fontWeight="semibold">
                {t('modelManager.starterBundles')}
              </Text>
              <Tooltip label={t('modelManager.starterBundleHelpText')}>
                <Box>
                  <Icon as={PiInfoBold} color="base.200" />
                </Box>
              </Tooltip>
            </Flex>
            <Flex gap={2}>
              {map(results.starter_bundles, (bundle, bundleName) => (
                <StarterBundle key={bundleName} bundleName={bundleName} bundle={bundle} />
              ))}
            </Flex>
          </Flex>
        )}
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
              <StarterModelsResultItem key={result.source} starterModel={result} />
            ))}
          </Flex>
        </ScrollableContent>
      </Flex>
    </Flex>
  );
});

StarterModelsResults.displayName = 'StarterModelsResults';
