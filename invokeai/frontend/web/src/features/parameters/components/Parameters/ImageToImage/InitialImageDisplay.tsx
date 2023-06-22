import { Flex, Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import InitialImagePreview from './InitialImagePreview';

const InitialImageDisplay = () => {
  const { t } = useTranslation();
  return (
    <Flex
      sx={{
        position: 'relative',
        flexDirection: 'column',
        height: '100%',
        width: '100%',
        rowGap: 4,
        borderRadius: 'base',
        bg: 'base.850',
        p: 4,
      }}
    >
      <Text
        sx={{
          px: 4,
          py: 2,
          color: 'base.300',
          backgroundColor: 'base.750',
          borderRadius: 4,
          fontSize: 14,
          fontWeight: 600,
          textAlign: 'center',
        }}
      >
        {t('parameters.initialImage')}
      </Text>
      <InitialImagePreview />
    </Flex>
  );
};

export default InitialImageDisplay;
