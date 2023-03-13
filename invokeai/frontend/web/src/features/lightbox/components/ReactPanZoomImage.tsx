import * as React from 'react';
import { TransformComponent, useTransformContext } from 'react-zoom-pan-pinch';
import * as InvokeAI from 'app/invokeai';

type ReactPanZoomProps = {
  image: InvokeAI.Image;
  styleClass?: string;
  alt?: string;
  ref?: React.Ref<HTMLImageElement>;
  rotation: number;
  scaleX: number;
  scaleY: number;
};

export default function ReactPanZoomImage({
  image,
  alt,
  ref,
  styleClass,
  rotation,
  scaleX,
  scaleY,
}: ReactPanZoomProps) {
  const { centerView } = useTransformContext();

  return (
    <TransformComponent
      wrapperStyle={{
        width: '100%',
        height: '100%',
      }}
    >
      <img
        style={{
          transform: `rotate(${rotation}deg) scaleX(${scaleX})  scaleY(${scaleY})`,
          width: '100%',
        }}
        src={image.url}
        alt={alt}
        ref={ref}
        className={styleClass ? styleClass : ''}
        onLoad={() => centerView(1, 0, 'easeOut')}
      />
    </TransformComponent>
  );
}
