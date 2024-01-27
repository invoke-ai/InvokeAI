import type { ChakraProps } from '@invoke-ai/ui-library';
import { Flex, FormControl, FormLabel, ListItem, OrderedList, Spinner, Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { selectDynamicPromptsSlice } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWarningCircleBold } from 'react-icons/pi';

const selectPrompts = createMemoizedSelector(selectDynamicPromptsSlice, (dynamicPrompts) => dynamicPrompts.prompts);

const listItemStyles: ChakraProps['sx'] = {
  '&::marker': { color: 'base.500' },
};

const ParamDynamicPromptsPreview = () => {
  const { t } = useTranslation();
  const parsingError = useAppSelector((s) => s.dynamicPrompts.parsingError);
  const isError = useAppSelector((s) => s.dynamicPrompts.isError);
  const isLoading = useAppSelector((s) => s.dynamicPrompts.isLoading);
  const prompts = useAppSelector(selectPrompts);

  const label = useMemo(() => {
    let _label = `${t('dynamicPrompts.promptsPreview')} (${prompts.length})`;
    if (parsingError) {
      _label += ` - ${parsingError}`;
    }
    return _label;
  }, [parsingError, prompts.length, t]);

  if (isError) {
    return (
      <Flex w="full" h="full" layerStyle="second" alignItems="center" justifyContent="center" p={8}>
        <IAINoContentFallback icon={PiWarningCircleBold} label="Problem generating prompts" />
      </Flex>
    );
  }

  return (
    <FormControl orientation="vertical" w="full" h="full" isInvalid={Boolean(parsingError || isError)}>
      <InformationalPopover feature="dynamicPrompts" inPortal={false}>
        <FormLabel>{label}</FormLabel>
      </InformationalPopover>
      <Flex w="full" h="full" pos="relative" layerStyle="first" p={4} borderRadius="base">
        <ScrollableContent>
          <OrderedList stylePosition="inside" ms={0}>
            {prompts.map((prompt, i) => (
              <ListItem fontSize="sm" key={`${prompt}.${i}`} sx={listItemStyles}>
                <Text as="span">{prompt}</Text>
              </ListItem>
            ))}
          </OrderedList>
        </ScrollableContent>
        {isLoading && (
          <Flex
            pos="absolute"
            w="full"
            h="full"
            top={0}
            insetInlineStart={0}
            layerStyle="second"
            opacity={0.7}
            alignItems="center"
            justifyContent="center"
          >
            <Spinner />
          </Flex>
        )}
      </Flex>
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsPreview);
