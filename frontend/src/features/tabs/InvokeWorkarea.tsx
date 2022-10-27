import { ReactNode } from 'react';
import { RootState, useAppSelector } from '../../app/store';
import ImageGallery from '../gallery/ImageGallery';
import ShowHideGalleryButton from '../gallery/ShowHideGalleryButton';

type InvokeWorkareaProps = {
  optionsPanel: ReactNode;
  className?: string;
  children: ReactNode;
};

const InvokeWorkarea = (props: InvokeWorkareaProps) => {
  const { optionsPanel, className, children } = props;

  const { shouldShowGallery } = useAppSelector(
    (state: RootState) => state.gallery
  );

  return (
    <div
      className={
        className ? `workarea-container ${className}` : `workarea-container`
      }
    >
      <div className="workarea">
        <div className="workarea-options-panel">{optionsPanel}</div>
        <div className="workarea-content">{children}</div>
        <ImageGallery />
      </div>
      {!shouldShowGallery && <ShowHideGalleryButton />}
    </div>
  );
};

export default InvokeWorkarea;
