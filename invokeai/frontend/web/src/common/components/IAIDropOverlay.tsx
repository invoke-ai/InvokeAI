import type { TextProps } from '@invoke-ai/ui-library';
import { Flex, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  isOver: boolean;
  label?: string;
  textStyleOverrides?: Partial<TextProps>;
};

const IAIDropOverlay = (props: Props) => {
  const { t } = useTranslation();
  const { isOver, textStyleOverrides, label = t('gallery.drop') } = props;
  return (
    <Flex position="absolute" top={0} right={0} bottom={0} left={0}>
      <Flex
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        w="full"
        h="full"
        bg="base.900"
        opacity={0.7}
        borderRadius="base"
        alignItems="center"
        justifyContent="center"
        transitionProperty="common"
        transitionDuration="0.1s"
      />

      <Flex
        position="absolute"
        top={0.5}
        right={0.5}
        bottom={0.5}
        left={0.5}
        opacity={1}
        borderWidth={1.5}
        borderColor={isOver ? 'invokeYellow.300' : 'base.500'}
        borderRadius="base"
        borderStyle="dashed"
        transitionProperty="common"
        transitionDuration="0.1s"
        alignItems="center"
        justifyContent="center"
      >
        <Text
          fontSize="lg"
          fontWeight="semibold"
          color={isOver ? 'invokeYellow.300' : 'base.500'}
          transitionProperty="common"
          transitionDuration="0.1s"
          textAlign="center"
          {...textStyleOverrides}
        >
          {label}
        </Text>
      </Flex>
    </Flex>
  );
};

export default memo(IAIDropOverlay);
