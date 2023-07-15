import { Flex, Spacer, Text } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { toggleSize } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';
import { MdOutlineSwapVert } from 'react-icons/md';
import ParamAspectRatio from './ParamAspectRatio';
import ParamHeight from './ParamHeight';
import ParamWidth from './ParamWidth';
import { activeTabNameSelector } from '../../../../ui/store/uiSelectors';

export default function ParamSize() {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const shouldFitToWidthHeight = useAppSelector(
    (state: RootState) => state.generation.shouldFitToWidthHeight
  );
  const activeTabName = useAppSelector(activeTabNameSelector);
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
        <IAIIconButton
          tooltip={t('ui.swapSizes')}
          aria-label={t('ui.swapSizes')}
          size="sm"
          icon={<MdOutlineSwapVert />}
          fontSize={20}
          isDisabled={
            activeTabName === 'img2img' ? !shouldFitToWidthHeight : false
          }
          onClick={() => dispatch(toggleSize())}
        />
      </Flex>
      <Flex gap={2} alignItems="center">
        <Flex gap={2} flexDirection="column" width="full">
          <ParamWidth
            isDisabled={
              activeTabName === 'img2img' ? !shouldFitToWidthHeight : false
            }
          />
          <ParamHeight
            isDisabled={
              activeTabName === 'img2img' ? !shouldFitToWidthHeight : false
            }
          />
        </Flex>
      </Flex>
    </Flex>
  );
}
