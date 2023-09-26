import { useColorModeValue, useToken } from '@chakra-ui/react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { Group, Rect, Text } from 'react-konva';
import { CanvasImage } from '../store/canvasTypes';

type IAICanvasImageErrorFallbackProps = {
  canvasImage: CanvasImage;
};
const IAICanvasImageErrorFallback = ({
  canvasImage,
}: IAICanvasImageErrorFallbackProps) => {
  const [errorColorLight, errorColorDark, fontColorLight, fontColorDark] =
    useToken('colors', ['gray.400', 'gray.500', 'base.700', 'base.900']);
  const errorColor = useColorModeValue(errorColorLight, errorColorDark);
  const fontColor = useColorModeValue(fontColorLight, fontColorDark);
  const { t } = useTranslation();
  return (
    <Group>
      <Rect
        x={canvasImage.x}
        y={canvasImage.y}
        width={canvasImage.width}
        height={canvasImage.height}
        fill={errorColor}
      />
      <Text
        x={canvasImage.x}
        y={canvasImage.y}
        width={canvasImage.width}
        height={canvasImage.height}
        align="center"
        verticalAlign="middle"
        fontFamily='"Inter Variable", sans-serif'
        fontSize={canvasImage.width / 16}
        fontStyle="600"
        text={t('common.imageFailedToLoad')}
        fill={fontColor}
      />
    </Group>
  );
};

export default memo(IAICanvasImageErrorFallback);
