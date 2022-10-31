import { ReactNode } from 'react';
import ImageGallery from '../gallery/ImageGallery';

type InvokeWorkareaProps = {
  optionsPanel: ReactNode;
  children: ReactNode;
  styleClass?: string;
};

const InvokeWorkarea = (props: InvokeWorkareaProps) => {
  const { optionsPanel, children, styleClass } = props;

  return (
    <div
      className={
        styleClass ? `workarea-wrapper ${styleClass}` : `workarea-wrapper`
      }
    >
      <div className="workarea-main">
        {optionsPanel}
        {children}
        <ImageGallery />
      </div>
    </div>
  );
};

export default InvokeWorkarea;
