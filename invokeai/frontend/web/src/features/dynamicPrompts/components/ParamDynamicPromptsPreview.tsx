import {
  ChakraProps,
  Flex,
  FormControl,
  FormLabel,
  ListItem,
  OrderedList,
  Spinner,
  Text,
} from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCircleExclamation } from 'react-icons/fa6';

const selector = createMemoizedSelector(stateSelector, (state) => {
  const { isLoading, isError, prompts, parsingError } = state.dynamicPrompts;

  return {
    prompts,
    parsingError,
    isError,
    isLoading,
  };
});

const listItemStyles: ChakraProps['sx'] = {
  '&::marker': { color: 'base.500', _dark: { color: 'base.500' } },
};

const ParamDynamicPromptsPreview = () => {
  const { t } = useTranslation();
  const { prompts, parsingError, isLoading, isError } =
    useAppSelector(selector);

  if (isError) {
    return (
      <IAIInformationalPopover feature="dynamicPrompts">
        <Flex
          w="full"
          h="full"
          layerStyle="second"
          alignItems="center"
          justifyContent="center"
          p={8}
        >
          <IAINoContentFallback
            icon={FaCircleExclamation}
            label="Problem generating prompts"
          />
        </Flex>
      </IAIInformationalPopover>
    );
  }

  return (
    <IAIInformationalPopover feature="dynamicPrompts">
      <FormControl isInvalid={Boolean(parsingError)}>
        <FormLabel
          whiteSpace="nowrap"
          overflow="hidden"
          textOverflow="ellipsis"
        >
          {t('dynamicPrompts.promptsPreview')} ({prompts.length})
          {parsingError && ` - ${parsingError}`}
        </FormLabel>
        <Flex
          h={64}
          pos="relative"
          layerStyle="third"
          borderRadius="base"
          p={2}
        >
          <ScrollableContent>
            <OrderedList stylePosition="inside" ms={0}>
              {prompts.map((prompt, i) => (
                <ListItem
                  fontSize="sm"
                  key={`${prompt}.${i}`}
                  sx={listItemStyles}
                >
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
    </IAIInformationalPopover>
  );
};

export default memo(ParamDynamicPromptsPreview);
