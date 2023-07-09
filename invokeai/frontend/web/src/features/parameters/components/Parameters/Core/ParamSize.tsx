import { Flex, Spacer, Text } from '@chakra-ui/react';
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
      sx={{
        gap: 2,
        p: 4,
        borderRadius: 4,
        flexDirection: 'column',
        w: 'full',
        bg: 'base.150',
        _dark: {
          bg: 'base.750',
        },
      }}
    >
      <Flex alignItems="center" gap={2}>
        <Text
          sx={{
            fontSize: 'sm',
            width: 'full',
            color: 'base.700',
            _dark: {
              color: 'base.300',
            },
          }}
        >
          {t('parameters.aspectRatio')}
        </Text>
        <Spacer />
        <ParamAspectRatio />
      </Flex>
      <Flex gap={2} flexDirection="column">
        <ParamWidth isDisabled={!shouldFitToWidthHeight} />
        <ParamHeight isDisabled={!shouldFitToWidthHeight} />
      </Flex>
    </Flex>
  );
}
