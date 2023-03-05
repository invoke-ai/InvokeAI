import { Flex, Spinner, Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

interface LoaderProps {
  showText?: boolean;
  text?: string;
}

// This component loads before the theme so we cannot use theme tokens here

const Loading = (props: LoaderProps) => {
  const { t } = useTranslation();
  const { showText = false, text = t('common.loadingInvokeAI') } = props;

  return (
    <Flex
      width="100vw"
      height="100vh"
      alignItems="center"
      justifyContent="center"
      bg="#121212"
      flexDirection="column"
      rowGap={4}
    >
      <Spinner color="grey" w="5rem" h="5rem" />
      {showText && (
        <Text
          color="grey"
          fontWeight="semibold"
          fontFamily="'Inter', sans-serif"
        >
          {text}
        </Text>
      )}
    </Flex>
  );
};

export default Loading;
