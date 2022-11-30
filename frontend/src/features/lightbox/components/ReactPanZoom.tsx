import IAIIconButton from 'common/components/IAIIconButton';
import * as React from 'react';
import {
  BiReset,
  BiRotateLeft,
  BiRotateRight,
  BiZoomIn,
  BiZoomOut,
} from 'react-icons/bi';
import { MdFlip } from 'react-icons/md';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

type ReactPanZoomProps = {
  image: string;
  styleClass?: string;
  alt?: string;
  ref?: any;
};

export default function ReactPanZoom({
  image,
  alt,
  ref,
  styleClass,
}: ReactPanZoomProps) {
  const [rotation, setRotation] = React.useState(0);
  const [flip, setFlip] = React.useState(false);

  const rotateLeft = () => {
    if (rotation === -3) {
      setRotation(0);
    } else {
      setRotation(rotation - 1);
    }
  };

  const rotateRight = () => {
    if (rotation === 3) {
      setRotation(0);
    } else {
      setRotation(rotation + 1);
    }
  };

  const flipImage = () => {
    setFlip(!flip);
  };

  return (
    <TransformWrapper
      centerOnInit
      minScale={0.1}
      initialPositionX={50}
      initialPositionY={50}
    >
      {({ zoomIn, zoomOut, resetTransform, centerView }) => (
        <>
          <div className="lightbox-image-options">
            <IAIIconButton
              icon={<BiZoomIn />}
              aria-label="Zoom In"
              tooltip="Zoom In"
              onClick={() => zoomIn()}
              fontSize={20}
            />

            <IAIIconButton
              icon={<BiZoomOut />}
              aria-label="Zoom Out"
              tooltip="Zoom Out"
              onClick={() => zoomOut()}
              fontSize={20}
            />

            <IAIIconButton
              icon={<BiRotateLeft />}
              aria-label="Rotate Left"
              tooltip="Rotate Left"
              onClick={rotateLeft}
              fontSize={20}
            />

            <IAIIconButton
              icon={<BiRotateRight />}
              aria-label="Rotate Right"
              tooltip="Rotate Right"
              onClick={rotateRight}
              fontSize={20}
            />

            <IAIIconButton
              icon={<MdFlip />}
              aria-label="Flip Image"
              tooltip="Flip Image"
              onClick={flipImage}
              fontSize={20}
            />

            <IAIIconButton
              icon={<BiReset />}
              aria-label="Reset"
              tooltip="Reset"
              onClick={() => {
                resetTransform();
                setRotation(0);
                setFlip(false);
              }}
              fontSize={20}
            />
          </div>
          <TransformComponent
            wrapperStyle={{
              width: '100%',
              height: '100%',
            }}
          >
            <img
              style={{
                transform: `rotate(${rotation * 90}deg) scaleX(${
                  flip ? -1 : 1
                })`,
                width: '100%',
              }}
              src={image}
              alt={alt}
              ref={ref}
              className={styleClass ? styleClass : ''}
              onLoad={() => centerView(1, 0, 'easeOut')}
            />
          </TransformComponent>
        </>
      )}
    </TransformWrapper>
  );
}
