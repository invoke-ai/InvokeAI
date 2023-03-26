import { ButtonGroup } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { useTranslation } from 'react-i18next';
import {
  BiReset,
  BiRotateLeft,
  BiRotateRight,
  BiZoomIn,
  BiZoomOut,
} from 'react-icons/bi';
import { MdFlip } from 'react-icons/md';
import { useTransformContext } from 'react-zoom-pan-pinch';

type ReactPanZoomButtonsProps = {
  flipHorizontally: () => void;
  flipVertically: () => void;
  rotateCounterClockwise: () => void;
  rotateClockwise: () => void;
  reset: () => void;
};

const ReactPanZoomButtons = ({
  flipHorizontally,
  flipVertically,
  rotateCounterClockwise,
  rotateClockwise,
  reset,
}: ReactPanZoomButtonsProps) => {
  const { zoomIn, zoomOut, resetTransform } = useTransformContext();
  const { t } = useTranslation();

  return (
    <ButtonGroup isAttached orientation="vertical">
      <IAIIconButton
        icon={<BiZoomIn />}
        aria-label={t('accessibility.zoomIn')}
        tooltip={t('accessibility.zoomIn')}
        onClick={() => zoomIn()}
        fontSize={20}
      />

      <IAIIconButton
        icon={<BiZoomOut />}
        aria-label={t('accessibility.zoomOut')}
        tooltip={t('accessibility.zoomOut')}
        onClick={() => zoomOut()}
        fontSize={20}
      />

      <IAIIconButton
        icon={<BiRotateLeft />}
        aria-label={t('accessibility.rotateCounterClockwise')}
        tooltip={t('accessibility.rotateCounterClockwise')}
        onClick={rotateCounterClockwise}
        fontSize={20}
      />

      <IAIIconButton
        icon={<BiRotateRight />}
        aria-label={t('accessibility.rotateClockwise')}
        tooltip={t('accessibility.rotateClockwise')}
        onClick={rotateClockwise}
        fontSize={20}
      />

      <IAIIconButton
        icon={<MdFlip />}
        aria-label={t('accessibility.flipHorizontally')}
        tooltip={t('accessibility.flipHorizontally')}
        onClick={flipHorizontally}
        fontSize={20}
      />

      <IAIIconButton
        icon={<MdFlip style={{ transform: 'rotate(90deg)' }} />}
        aria-label={t('accessibility.flipVertically')}
        tooltip={t('accessibility.flipVertically')}
        onClick={flipVertically}
        fontSize={20}
      />

      <IAIIconButton
        icon={<BiReset />}
        aria-label={t('accessibility.reset')}
        tooltip={t('accessibility.reset')}
        onClick={() => {
          resetTransform();
          reset();
        }}
        fontSize={20}
      />
    </ButtonGroup>
  );
};

export default ReactPanZoomButtons;
