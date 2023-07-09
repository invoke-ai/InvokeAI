import { Flex, Text } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useTranslation } from 'react-i18next';
import ParamAspectRatio from './ParamAspectRatio';
import ParamHeight from './ParamHeight';
import ParamWidth from './ParamWidth';

export default function ParamSize() {
  const { t } = useTranslation();
  const shouldFitToWidthHeight = useAppSelector(
    (state: RootState) => state.generation.shouldFitToWidthHeight
  );
  return (
    <Flex
      gap={2}
      bg="base.900"
      p={4}
      borderRadius={4}
      flexDirection="column"
      w="100%"
    >
      <Flex alignItems="center" gap={2}>
        <Text fontSize={14} width="full" color="base.300">
          {t('parameters.aspectRatio')}
        </Text>
        <ParamAspectRatio />
      </Flex>
      <Flex gap={2} flexDirection="column">
        <ParamWidth isDisabled={!shouldFitToWidthHeight} />
        <ParamHeight isDisabled={!shouldFitToWidthHeight} />
      </Flex>
    </Flex>
  );
}
