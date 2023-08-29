import { Flex, Spacer, Text } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { flipBoundingBoxAxes } from 'features/canvas/store/canvasSlice';
import { useTranslation } from 'react-i18next';
import { MdOutlineSwapVert } from 'react-icons/md';
import ParamAspectRatio from '../../Core/ParamAspectRatio';
import ParamBoundingBoxHeight from './ParamBoundingBoxHeight';
import ParamBoundingBoxWidth from './ParamBoundingBoxWidth';

export default function ParamBoundingBoxSize() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  return (
    <Flex
      sx={{
        gap: 2,
        p: 4,
        borderRadius: 4,
        flexDirection: 'column',
        w: 'full',
        bg: 'base.100',
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
        <IAIIconButton
          tooltip={t('ui.swapSizes')}
          aria-label={t('ui.swapSizes')}
          size="sm"
          icon={<MdOutlineSwapVert />}
          fontSize={20}
          onClick={() => dispatch(flipBoundingBoxAxes())}
        />
      </Flex>
      <ParamBoundingBoxWidth />
      <ParamBoundingBoxHeight />
    </Flex>
  );
}
