import * as React from 'react';
import { TransformComponent, useTransformContext } from 'react-zoom-pan-pinch';
import { useGetUrl } from 'common/util/getUrl';
import { ImageDTO } from 'services/api';

type ReactPanZoomProps = {
  image: ImageDTO;
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
  const { getUrl } = useGetUrl();

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
        src={getUrl(image.image_url)}
        alt={alt}
        ref={ref}
        className={styleClass ? styleClass : ''}
        onLoad={() => centerView(1, 0, 'easeOut')}
      />
    </TransformComponent>
  );
}
