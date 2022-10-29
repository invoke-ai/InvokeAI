import { ReactNode } from 'react';
import { RootState, useAppSelector } from '../../app/store';
import ImageGallery from '../gallery/ImageGallery';
import ShowHideGalleryButton from '../gallery/ShowHideGalleryButton';

type InvokeWorkareaProps = {
  optionsPanel: ReactNode;
  children: ReactNode;
  styleClass?: string;
};

const InvokeWorkarea = (props: InvokeWorkareaProps) => {
  const { optionsPanel, children, styleClass } = props;

  const { shouldShowGallery, shouldHoldGalleryOpen, shouldPinGallery } =
    useAppSelector((state: RootState) => state.gallery);

  return (
    <div
      className={
        styleClass ? `workarea-wrapper ${styleClass}` : `workarea-wrapper`
      }
    >
      <div className="workarea-main">
        <div className="workarea-options-panel">{optionsPanel}</div>
        {children}
        <ImageGallery />
      </div>
      {!(shouldShowGallery || (shouldHoldGalleryOpen && !shouldPinGallery)) && (
        <ShowHideGalleryButton />
      )}
    </div>
  );
};

export default InvokeWorkarea;
