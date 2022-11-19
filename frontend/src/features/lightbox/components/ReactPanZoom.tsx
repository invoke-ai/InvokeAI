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
import { PanViewer } from 'react-image-pan-zoom-rotate';

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
  const [dx, setDx] = React.useState(0);
  const [dy, setDy] = React.useState(0);
  const [zoom, setZoom] = React.useState(1);
  const [rotation, setRotation] = React.useState(0);
  const [flip, setFlip] = React.useState(false);

  const resetAll = () => {
    setDx(0);
    setDy(0);
    setZoom(1);
    setRotation(0);
    setFlip(false);
  };
  const zoomIn = () => {
    setZoom(zoom + 0.2);
  };

  const zoomOut = () => {
    if (zoom >= 0.5) {
      setZoom(zoom - 0.2);
    }
  };

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

  const onPan = (dx: number, dy: number) => {
    setDx(dx);
    setDy(dy);
  };

  return (
    <div>
      <div className="lightbox-image-options">
        <IAIIconButton
          icon={<BiZoomIn />}
          aria-label="Zoom In"
          tooltip="Zoom In"
          onClick={zoomIn}
          fontSize={20}
        />

        <IAIIconButton
          icon={<BiZoomOut />}
          aria-label="Zoom Out"
          tooltip="Zoom Out"
          onClick={zoomOut}
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
          onClick={resetAll}
          fontSize={20}
        />
      </div>
      <PanViewer
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1,
        }}
        zoom={zoom}
        setZoom={setZoom}
        pandx={dx}
        pandy={dy}
        onPan={onPan}
        rotation={rotation}
        key={dx}
      >
        <img
          style={{
            transform: `rotate(${rotation * 90}deg) scaleX(${flip ? -1 : 1})`,
            width: '100%',
          }}
          src={image}
          alt={alt}
          ref={ref}
          className={styleClass ? styleClass : ''}
        />
      </PanViewer>
    </div>
  );
}
