import { useToken } from '@chakra-ui/react';
import type { CanvasImage } from 'features/canvas/store/canvasTypes';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { Group, Rect, Text } from 'react-konva';

type IAICanvasImageErrorFallbackProps = {
  canvasImage: CanvasImage;
};
const IAICanvasImageErrorFallback = ({
  canvasImage,
}: IAICanvasImageErrorFallbackProps) => {
  const [rectFill, textFill] = useToken('colors', ['base.500', 'base.900']);
  const { t } = useTranslation();
  return (
    <Group>
      <Rect
        x={canvasImage.x}
        y={canvasImage.y}
        width={canvasImage.width}
        height={canvasImage.height}
        fill={rectFill}
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
        fill={textFill}
      />
    </Group>
  );
};

export default memo(IAICanvasImageErrorFallback);
